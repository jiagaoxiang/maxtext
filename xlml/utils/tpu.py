# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to create, delete, and SSH with TPUs."""

import datetime
import io
import itertools
import os
from typing import Dict, Iterable, Optional, Tuple
import uuid

from absl import logging
import airflow
from airflow.decorators import task, task_group
from airflow.utils.task_group import TaskGroup
from airflow.operators.python import get_current_context
from xlml.apis import gcp_config, test_config
import fabric
import google.api_core.exceptions
import google.auth
import google.cloud.tpu_v2alpha1 as tpu_api
import google.longrunning.operations_pb2 as operations
from xlml.utils import ssh, startup_script
import paramiko
from airflow.models import Variable
from google.protobuf.duration_pb2 import Duration


@task
def generate_tpu_name(
    base_tpu_name: str,
    set_env_var: bool,
) -> str:
  tpu_name = f'{base_tpu_name}-{str(uuid.uuid4())}'
  if set_env_var:
    Variable.set(base_tpu_name, tpu_name)
  return tpu_name


def create_queued_resource(
    tpu_name: airflow.XComArg,
    gcp: gcp_config.GCPConfig,
    ssh_keys: airflow.XComArg,
    timeout: datetime.timedelta,
    task_test_config: test_config,
    use_startup_script: bool = False,
) -> Tuple[TaskGroup, airflow.XComArg]:
  """Request a QueuedResource and wait until the nodes are created.

  Args:
    tpu_name: XCom value for unique TPU name.
    accelerator: Description of TPU to create.
    gcp: GCP project/zone configuration.
    ssh_keys: XCom value for SSH keys to communicate with these TPUs.
    timeout: Amount of time to wait for TPUs to be created.
    num_slices: Number of TPU slices.

  Returns:
    A TaskGroup for the entire create operation and an XCom value for the
    qualified queued_resource name.
  """

  @task
  def create_queued_resource_request(tpu_name: str, ssh_keys: ssh.SshKeys) -> str:
    creds, _ = google.auth.default()
    client = tpu_api.TpuClient(credentials=creds)

    parent = f'projects/{gcp.project_name}/locations/{gcp.zone}'

    # Determine node_id and multiNodeParams based on num_slices
    if task_test_config.num_slices == 1:
      node_id = tpu_name
      multi_node_params = None
    else:
      node_id = None
      multi_node_params = tpu_api.types.QueuedResource.Tpu.NodeSpec.MultiNodeParams(
          node_count=task_test_config.num_slices, node_id_prefix=tpu_name
      )

    startup_script_command = ''

    if use_startup_script:
      main_command = '\n'.join(
          task_test_config.set_up_cmds + task_test_config.run_model_cmds
      )
      startup_script_command = startup_script.generate_startup_script(main_command)

    metadata = {
        'ssh-keys': f'ml-auto-solutions:{ssh_keys.public}',
        'startup-script': startup_script_command,
    }

    queued_resource = tpu_api.QueuedResource(
        # TODO(ranran): enable configuration via `AcceleratorConfig`
        tpu=tpu_api.QueuedResource.Tpu(
            node_spec=[
                tpu_api.QueuedResource.Tpu.NodeSpec(
                    node_id=node_id,
                    multi_node_params=multi_node_params,
                    parent=parent,
                    node=tpu_api.Node(
                        accelerator_type=task_test_config.accelerator.name,
                        description='noteardown',
                        runtime_version=task_test_config.accelerator.runtime_version,
                        network_config=tpu_api.NetworkConfig(
                            network=task_test_config.accelerator.network,
                            subnetwork=task_test_config.accelerator.subnetwork,
                            enable_external_ips=True,
                        ),
                        metadata=metadata,
                    ),
                )
            ],
        ),
        guaranteed=tpu_api.QueuedResource.Guaranteed(
            reserved=task_test_config.accelerator.reserved,
        ),
        queueing_policy=tpu_api.QueuedResource.QueueingPolicy(
            valid_until_duration=Duration(seconds=int(timeout.total_seconds())),
        ),
    )

    qr_operation = client.create_queued_resource(
        parent=parent,
        queued_resource_id=tpu_name,
        queued_resource=queued_resource,
    )
    response = qr_operation.result()
    logging.info('Create QR response: {}'.format(response))
    # TODO(wcromar): do anything about failures

    return response.name

  @task.sensor(poke_interval=60, timeout=timeout.total_seconds(), mode='reschedule')
  def wait_for_ready_queued_resource(qualified_name: str):
    creds, _ = google.auth.default()
    client = tpu_api.TpuClient(credentials=creds)

    qr = client.get_queued_resource(name=qualified_name)
    state = qr.state.state
    logging.info(f'Queued resource state: {state.name}')
    if qr.state.state == tpu_api.QueuedResourceState.State.ACTIVE:
      return True
    elif qr.state.state in [
        tpu_api.QueuedResourceState.State.CREATING,
        tpu_api.QueuedResourceState.State.WAITING_FOR_RESOURCES,
        tpu_api.QueuedResourceState.State.ACCEPTED,
        tpu_api.QueuedResourceState.State.PROVISIONING,
    ]:
      return False
    else:
      raise RuntimeError(f'Bad queued resource state {state.name}')

  def check_if_startup_script_end(
      queued_resource: airflow.XComArg, ssh_keys: airflow.XComArg
  ):
    check_script = startup_script.monitor_startup_script()

    return ssh_tpu.override(
        task_id='check_if_startup_script_end',
        execution_timeout=datetime.timedelta(minutes=task_test_config.time_out_in_min),
        owner=task_test_config.task_owner,
    )(
        queued_resource,
        check_script,
        ssh_keys,
        False,
    )

  with TaskGroup(group_id='create_queued_resource') as tg:
    qualified_name = create_queued_resource_request(tpu_name, ssh_keys)

    if use_startup_script:
      wait_for_ready_queued_resource(qualified_name) >> check_if_startup_script_end(
          qualified_name, ssh_keys
      )
    else:
      wait_for_ready_queued_resource(qualified_name)

  return tg, qualified_name


@task_group
def delete_queued_resource(qualified_name: airflow.XComArg):
  """Implements cascading delete for a Queued Resource.

  Args:
    qualified_name: XCom value holding the qualified name of the queued
      resource.
  """

  @task(trigger_rule='all_done')
  def delete_tpu_nodes_request(qualified_name: str):
    # TODO(wcromar): Find a less repetitive way to manage the TPU client.
    creds, _ = google.auth.default()
    client = tpu_api.TpuClient(credentials=creds)

    try:
      qr = client.get_queued_resource(name=qualified_name)
    except google.api_core.exceptions.NotFound:
      logging.info(f'{qualified_name} not found')
      return

    for node in qr.tpu.node_spec:
      try:
        op = client.delete_node(name=f'{node.parent}/nodes/{node.node_id}')
        logging.info('Delete node state: {}'.format(op))
      except google.api_core.exceptions.NotFound:
        logging.info(f'{node.node_id} is already deleted')

  @task.sensor(poke_interval=60, timeout=3600, mode='reschedule')
  def wait_for_tpu_deletion(qualified_name: str):
    creds, _ = google.auth.default()
    client = tpu_api.TpuClient(credentials=creds)

    qr = client.get_queued_resource(name=qualified_name)
    # Queued Resources can only be deleted once they are SUSPENDED, even if all
    # underlying nodes have already been deleted.
    if qr.state.state in [
        tpu_api.QueuedResourceState.State.SUSPENDED,
        # TPU will be sitting in WAITING_FOR_RESOURCES if creation timed out.
        tpu_api.QueuedResourceState.State.WAITING_FOR_RESOURCES,
        tpu_api.QueuedResourceState.State.ACCEPTED,
    ]:
      logging.info(f'All TPU nodes deleted for {qualified_name}')
      return True

    logging.info(f'TPU Nodes: {qr.tpu.node_spec}')
    return False

  @task(trigger_rule='all_done')
  def delete_queued_resource_request(qualified_name: str) -> Optional[str]:
    creds, _ = google.auth.default()
    client = tpu_api.TpuClient(credentials=creds)

    try:
      op = client.delete_queued_resource(name=qualified_name)
      logging.info(f'delete op {op}')
    except google.api_core.exceptions.NotFound:
      logging.info(f'{qualified_name} is already deleted')
      return None

    return op.operation.name

  @task.sensor(poke_interval=60, timeout=3600, mode='reschedule')
  def wait_for_queued_resource_deletion(op_name: Optional[str]):
    if not op_name:
      logging.info('No delete operation given')
      return True

    creds, _ = google.auth.default()
    client = tpu_api.TpuClient(credentials=creds)

    op = client.get_operation(operations.GetOperationRequest(name=op_name))
    return op.done

  delete_tpu_nodes = delete_tpu_nodes_request(qualified_name) >> wait_for_tpu_deletion(
      qualified_name
  )
  qr_op_name = delete_tpu_nodes >> delete_queued_resource_request(qualified_name)
  wait_for_queued_resource_deletion(qr_op_name)


def kill_process_by_pid() -> str:
  return f"""accelerator_type=\${{1}}
  if [[ \${{accelerator_type}} =~ ^v5.* ]] 
  then
    device_name=vfio/*
  else 
    device_name=accel*
  fi
  echo \\"Terminating all processes utilizing the TPU (if any).\\"
  sudo lsof -t /dev/\${{device_name}} | xargs -r kill -9
  """


@task
def ssh_tpu(
    qualified_name: str,
    cmds: Iterable[str],
    ssh_keys: ssh.SshKeys,
    all_workers: bool,
    env: Dict[str, str] = None,
) -> None:
  """SSH TPU and run commands in multi process.

  Args:
   qualified_name: The qualified name of a queued resource.
   cmds: The commands to run on a TPU.
   ssh_keys: The SSH key pair to use for authentication.
   all_workers: The flag to define if run commands on all workers or worker 0
     only.
   env: environment variables to be pass to the ssh runner session using dict.
  """
  creds, _ = google.auth.default()
  client = tpu_api.TpuClient(credentials=creds)

  queued_resource = client.get_queued_resource(name=qualified_name)

  nodes = [
      client.get_node(name=os.path.join(node.parent, 'nodes', node.node_id))
      for node in queued_resource.tpu.node_spec
  ]

  if all_workers:
    endpoints = itertools.chain.from_iterable(node.network_endpoints for node in nodes)
    ip_addresses = [endpoint.ip_address for endpoint in endpoints]
    logging.info(f'Connecting to IP addresses of all workers: {ip_addresses}')
  else:
    ip_addresses = [nodes[0].network_endpoints[0].ip_address]
    logging.info(f'Connecting to IP addresses of worker 0: {ip_addresses}')

  pkey = paramiko.RSAKey.from_private_key(io.StringIO(ssh_keys.private))
  ssh_group = fabric.ThreadingGroup(
      *ip_addresses,
      connect_kwargs={
          'auth_strategy': paramiko.auth_strategy.InMemoryPrivateKey(
              'ml-auto-solutions', pkey
          )
      },
  )

  context = get_current_context()
  if context['task_instance'].try_number > 1:
    # kill TPU process by pid (if any) to avoid `TPU in use` error in retry
    tmp_file = '/tmp/kill_process.sh'
    accelerator_type = nodes[0].accelerator_type
    script = kill_process_by_pid()
    kill_process_cmds = (
        f'set -xue; sudo echo "{script}" > {tmp_file}',
        f'bash {tmp_file} {accelerator_type}',
    )
    ssh_group.run(';'.join(kill_process_cmds))

  # run provided commands
  ssh_group.run(cmds, env=env)


@task
def clean_up_idle_queued_resources(project_name: str, zones: Iterable[str]) -> None:
  """Clean up queued resources in FAILED or SUSPENDED states.

  Args:
   project_name: The project of resources.
   zones: Available zones to clean up for the project.
  """
  creds, _ = google.auth.default()
  client = tpu_api.TpuClient(credentials=creds)

  logging.info(f'Cleaning up resources in project {project_name}.')
  for zone in zones:
    logging.info(f'Checking in zone {zone.value}.')
    parent = f'projects/{project_name}/locations/{zone.value}'
    request = tpu_api.types.ListQueuedResourcesRequest(parent=parent)
    responses = client.list_queued_resources(request)

    for qr in responses:
      state = qr.state.state
      if (
          state == tpu_api.QueuedResourceState.State.FAILED
          or state == tpu_api.QueuedResourceState.State.SUSPENDED
      ):
        logging.info(f'Deleting {qr.name} in {state.name} status.')
        client.delete_queued_resource(name=qr.name)
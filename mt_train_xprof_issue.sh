#!/bin/bash

export MODEL_NAME=llama2-7b

# Launch llama2 70b
# export MODEL_NAME=llama2-70b

# Launch llama3.1 405b
# export MODEL_NAME=llama3.1-405b

# Common parameters
export CLUSTER_NAME=a3plus-benchmark

# export ZONE=us-central1-b
export ZONE=australia-southeast1

export DEVICE_TYPE=h100-mega-80gb-8
export OUTPUT_PATH=lancewang-dev-supercomputer-testing/maxtext_gpu
export OUTPUT_BUCKET=gs://$OUTPUT_PATH
export RUN_NAME=maxtext-$MODEL_NAME
CONFIG_NAME=$(echo $MODEL_NAME | sed 's/-/_/g')

# export JAX_ENABLE_PGLE=false
export JAX_ENABLE_PGLE=false
export JAX_PGLE_AGGREGATION_PERCENTILE=50
export JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=true
export JAX_PGLE_PROFILING_RUNS=3
export STRICT_CHECKER=false
# JAX_PGLE_AGGREGATION_PERCENTILE=$JAX_PGLE_AGGREGATION_PERCENTILE
# JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS=$JAX_SHARE_AUTOTUNE_CONFIG_BETWEEN_HOSTS
# JAX_PGLE_PROFILING_RUNS=$JAX_PGLE_PROFILING_RUNS
# TF_CPP_VMODULE=profile_guided_latency_estimator=10
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
# CUDA_DEVICE_MAX_CONNECTIONS=1

cat <<EOF > env.txt
TF_CPP_MIN_LOG_LEVEL=0
TF_CPP_VMODULE=gpu_executable=2
NCCL_DEBUG=INFO
NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/usr/local/nvidia/lib64/a3plus_guest_config.textproto
NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
JAX_ENABLE_PGLE=$JAX_ENABLE_PGLE
JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=$JAX_ENABLE_PGLE
JAX_DEBUG_LOG_MODULES=compiler
XLA_FLAGS=--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_triton_gemm=false \
--xla_gpu_enable_highest_priority_async_stream=true \
--xla_gpu_all_reduce_combine_threshold_bytes=536870912 \
--xla_gpu_all_gather_combine_threshold_bytes=536870912 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=536870912 \
--xla_gpu_enable_pipelined_all_gather=true \
--xla_gpu_enable_pipelined_reduce_scatter=true \
--xla_gpu_enable_pipelined_all_reduce=true \
--xla_gpu_enable_while_loop_double_buffering=true \
--xla_disable_hlo_passes=rematerialization \
--xla_gpu_enable_pgle_accuracy_checker=$STRICT_CHECKER \
--xla_gpu_enable_triton_softmax_fusion=false \
--xla_gpu_enable_all_gather_combine_by_dim=false \
--xla_gpu_enable_reduce_scatter_combine_by_dim=false
EOF

call_config() {
    new_remat_name=$(echo $6| awk -F'_' '{for (i=1; i<=NF; i++) printf "%s", substr($i,1,1); print ""}')
    export WORKLOAD_NAME=$USER-405b-$1n$2b$3tp$4pp$5fsdp$6l-$new_remat_name-err

    export NUM_NODES=$1

    export PER_DEVICE_BATCH_SIZE=$2
    export ICI_TP=$3
    # export ICI_TP=1

    export DCN_PP=$4
    # The rest goes to DP
    export DCN_FSDP=$5

    export NUM_LAYERS_PER_PP_STAGE=$6 # Layers are modified to 128 for short term solution
    # export NUM_LAYERS_PER_PP_STAGE=$(expr 126 / $DCN_PP)

    export DCN_FSDP=$(expr $NUM_NODES / $DCN_PP)
    #export DCN_FSDP=32

    export REMAT_POLICY=$7
    # export REMAT_POLICY=full
    # export REMAT_POLICY=minimal

    export ATTENTION=cudnn_flash_te
    # export ATTENTION=dot_product
    export ICI_PP=$8
}

submit(){
    # export LOCAL_IMAGE_NAME=gcr.io/tpu-prod-env-multipod/jonbolin-maxtext-gpu:20241008-1
    # export LOCAL_IMAGE_NAME=us-west1-docker.pkg.dev/supercomputer-testing/lancewang/llama2-xprof_1010_nolayers_nightly_lance

    # Training with 8192 sequence length
    COMMAND="python3 MaxText/train.py MaxText/configs/models/gpu/$CONFIG_NAME.yml hardware=gpu run_name=$RUN_NAME steps=10 max_target_length=8192 model_name=$MODEL_NAME enable_checkpointing=false attention=$ATTENTION dataset_type=synthetic async_checkpointing=false base_output_directory=$OUTPUT_BUCKET logits_dot_in_fp32=false use_iota_embed=true ici_tensor_parallelism=$ICI_TP ici_pipeline_parallelism=$ICI_PP dcn_fsdp_parallelism=$DCN_FSDP dcn_pipeline_parallelism=$DCN_PP per_device_batch_size=$PER_DEVICE_BATCH_SIZE num_layers_per_pipeline_stage=$NUM_LAYERS_PER_PP_STAGE   weight_dtype=bfloat16 remat_policy=$REMAT_POLICY profiler=xplane skip_first_n_steps_for_profiler=5 "; 

    COMMAND='export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH;'"${COMMAND}"; 

    xpk workload delete --zone $ZONE --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME; 
    
    xpk workload create --zone $ZONE --cluster $CLUSTER_NAME --workload $WORKLOAD_NAME --command "${COMMAND}" --docker-image=$LOCAL_IMAGE_NAME --device-type=$DEVICE_TYPE --num-nodes=$NUM_NODES --priority=high --scheduler=gke.io/topology-aware-auto --env-file=env.txt ;
}

# input 1: 512 nodes 6k cluster
# input 2: per device batch size
# input 3: ICI TP
# input 4: DCN PP
# input 5: DCN FSDP
# input 5: number of layers per PP stage
# input 6:remat policy
# input 7: ICI PP 
call_config 1 1 1 1 1 8 save_qkv_proj 4
submit

# Best config, 
# call_config 512 1 1 1 512 0 save_qkv_proj
# submit 126

# Config 1, PP, FSDP 16 remat full, OOM
# call_config 512 3 8 2 16 16 full
# submit 128

# OOM
# call_config 512 3 8 2 256 16 full
# submit 128

# Congig 2, PP, OOM
# call_config 512 1 8 2 256 16 save_qkv_proj
# submit 128


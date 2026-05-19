export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export RAY_JOB_CONFIG_JSON_ENV_VAR='{
  "runtime_env": {
    "working_dir": "/workspace/verl",
      "env_vars": {
        "PYTHONPATH": "/workspace/Megatron-LM/:/workspace/mbridge/:$PYTHONPATH",
        "TOKENIZERS_PARALLELISM": "true",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_ORDER": "OAM_ID",
        "CUDART_DUMMY_REGISTER": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES":"1",
        "VLLM_USE_V1": "1",
        "VLLM_LOGGING_LEVEL": "DEBUG",
        "VERL_LOGGING_LEVEL": "DEBUG",
        "WANDB_KEY":"YOUR WANDB KEY",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN":"1",
        "BKCL_USE_AR": "1",
        "BKCL_RING_OPT": "1",
        "BKCL_FLAT_RING": "1",
        "BKCL_CCIX_RING": "1",
        "BKCL_TREE_THRESHOLD": "1",
        "BKCL_CCIX_BUFFER_GM": "1",
        "BKCL_FORCE_L3_RDMA": "0",
        "BKCL_RING_BUFFER_GM": "1",
        "BKCL_ENABLE_XDR": "1",
        "BKCL_RDMA_FORCE_TREE": "1",
        "BKCL_TREE_THRESHOLD": "1",
        "BKCL_XLINK_D2D": "0",
        "BKCL_XLINK_ETH": "0",
        "BKCL_XLINK_C2C": "1",
        "ALLREDUCE_ASYNC": "false",
        "ALLGATHER_ASYNC": "false",
        "ALLREDUCE_FUSION": "0",
        "XPU_FORCE_SHARED_DEVICE_CONTEXT": "1",
        "BKCL_RDMA_PROXY_DISABLE": "1",
        "BKCL_TRANS_UNSUPPORTED_DATATYPE": "1",
        "BKCL_KL3_TURBO_MODE": "1",
        "BKCL_RING_BUFFER_SIZE": "2097152",
        "BKCL_TIMEOUT": "400000",
        "CUDA_DISABLE_PRINTF": "1",
        "BKCL_RDMA_VERBS": "1",
        "XMLIR_FA_GEMM_TYPE": "float",
        "XBLAS_FC_HBM_VERSION": "40",
        "XMLIR_PARALLEL_SAVE_MEMORY": "false",
        "XMLIR_DISABLE_CUDA_ALLOCATOR": "false",
        "XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL": "0",
        "XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL": "False",
        "XMLIR_DUMP_FALLBACK_OP_LIST_BOOL": "true",
        "XMLIR_DIST_ASYNC_ISEND_IRECV": "false",
        "XMLIR_BATCH_PARALLEL": "false",
        "ENABLE_VLLM_XPU_CPU_BINDING": "all",
        "XPU_FORCE_USERMODE_LAUNCH": "1",
        "XMLIR_DIST_SINGLETON_STREAM": "true"
        
    }
  }
}'

set -x
ENGINE=${1:-vllm}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/geo3k/train.parquet \
    data.val_files=/workspace/geo3k/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=/workspace/models/Qwen3-VL-8B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.max_model_len=12800 \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.dtype="float16" \
    ++actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen3_vl_8b_fsdp' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
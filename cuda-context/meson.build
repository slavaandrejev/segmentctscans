cuda_context_lib = static_library(
    'find_intensity_range'
  , 'cuda-context.cpp'
  , include_directories: project_inc
  , dependencies: [
        cuda_dep
      , cccl_dep
      ]
  , cpp_args: [cuda_cpp_args]
  , build_by_default: false
  , override_options: cuda_overr_options
  )

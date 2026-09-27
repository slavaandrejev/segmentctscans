grad_magn_lib = static_library(
    'grad_magn'
  , 'grad_magn.cpp'
  , include_directories: project_inc
  , dependencies: [
        cuda_dep
      , cccl_dep
      ]
  , cpp_args: [cuda_cpp_args]
  , build_by_default: false
  , override_options: cuda_overr_options
  )

grad_magn_dep = declare_dependency(
    include_directories: project_inc
  , link_whole: [
        grad_magn_lib
      ]
  )

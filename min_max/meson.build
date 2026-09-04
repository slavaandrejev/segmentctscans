# takes too long in debug mode, always opimize this
min_max_lib = static_library(
    'min_max'
  , 'min_max.cpp'
  , include_directories: project_inc
  , dependencies: [
        cuda_dep
      , cccl_dep
      ]
  , cpp_args: [cuda_cpp_args]
  , build_by_default: false
  , override_options: {
        'optimization' : '3'
      } + cuda_overr_options
  )

min_max_dep = declare_dependency(
    include_directories: project_inc
  , link_whole: [
        min_max_lib
      ]
  )

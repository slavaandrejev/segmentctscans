# always compile IO with optimization, otherwise it is too slow
field_io_lib = static_library(
    'field_io'
  , 'field-io.cpp'
  , include_directories: project_inc
  , build_by_default: false
  , override_options: {
        'optimization' : '3'
      }
  )

uv_hist_lib = static_library(
    'uv_hist'
  , 'uv-hist-kernel.cpp'
  , include_directories: project_inc
  , dependencies: [
        cuda_dep
      , cccl_dep
      ]
  , cpp_args: [cuda_cpp_args]
  , build_by_default: false
  , override_options: cuda_overr_options
  )

uv_hist_dep = declare_dependency(
    include_directories: project_inc
  , link_whole: [
        uv_hist_lib
      ]
  , dependencies: [min_max_dep]
  )

executable(
    'uv-hist'
  , 'uv-hist.cpp'
  , include_directories: project_inc
  , link_whole: [
        field_io_lib
      , cuda_context_lib
      ]
  , dependencies: [
        boost_dep
      , fmt_dep
      , grad_magn_dep
      , mdspan_dep
      , omp_dep
      , opencv_dep
      , range_dep
      , uv_hist_dep
      ]
  )

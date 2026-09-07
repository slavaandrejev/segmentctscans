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

grad_magn_lib = static_library(
    'grad_magn'
  , 'grad_magn.cpp'
  , include_directories: project_inc
  , dependencies: [
        boost_dep
      , cuda_dep
      , cccl_dep
      ]
  , cpp_args: [cuda_cpp_args]
  , build_by_default: false
  , override_options: cuda_overr_options
  )

executable(
    'bimm-peaks'
  , 'bimm-peaks.cpp'
  , include_directories: project_inc
  , link_whole: [
        field_io_lib
      , cuda_context_lib
      , grad_magn_lib
      ]
  , dependencies: [
        boost_dep
      , fmt_dep
      , gsl_dep
      , libcmaes_dep
      , mdspan_dep
      , omp_dep
      , opencv_dep
      , range_dep
      ]
  )

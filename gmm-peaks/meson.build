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

executable(
    'gmm-peaks'
  , 'gmm-peaks.cpp'
  , include_directories: project_inc
  , link_whole: [
        field_io_lib
      , cuda_context_lib
      ]
  , dependencies: [
        boost_dep
      , calc_hist_dep
      , eigen_dep
      , fmt_dep
      , mdspan_dep
      , omp_dep
      , range_dep
      ])

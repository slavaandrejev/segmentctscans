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
    'totalvarmin'
  , 'totalvarmin.cpp'
  , include_directories: project_inc
  , link_whole: [field_io_lib]
  , dependencies: [
        boost_dep
      , clarabel_dep
      , fmt_dep
      , libtiff_dep
      , mdspan_dep
      , omp_dep
      , opencv_dep
      , range_dep
      ])

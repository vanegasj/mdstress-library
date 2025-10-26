# test/CMakeLists.txt

# Find all test soruce files that match test_*.cpp
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")

# Loop over each test file and create an executable
foreach(TEST_SRC ${TEST_SOURCES})
    # Get the filename without path and extension
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SRC})
   
    # Include headers
    target_include_directories(${TEST_NAME} PRIVATE
	${PROJECT_SOURCE_DIR}/include
    )

    # Link against mdstress
    target_link_libraries(${TEST_NAME} PRIVATE mdstress)

    # Enable warnings
    target_compile_options(${TEST_NAME} PRIVATE -Wall -Wextra -Werror)
endforeach()
	    

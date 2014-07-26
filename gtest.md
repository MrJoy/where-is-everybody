Suppose you put Google Test in directory ${GTEST_DIR}.  To build it,
create a library build target (or a project as called by Visual Studio
and Xcode) to compile


    ${GTEST_DIR}/src/gtest-all.cc

with

    ${GTEST_DIR}/include and ${GTEST_DIR}

in the header search path.  Assuming a Linux-like system and gcc,
something like the following will do:

    g++ -I${GTEST_DIR}/include -I${GTEST_DIR} -c ${GTEST_DIR}/src/gtest-all.cc
    ar -rv libgtest.a gtest-all.o

Next, you should compile your test source file with
${GTEST_DIR}/include in the header search path, and link it with gtest
and any other necessary libraries:

    g++ -I${GTEST_DIR}/include path/to/your_test.cc libgtest.a -o your_test


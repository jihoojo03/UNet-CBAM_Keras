# PCTest : Multi-processed Programming Assignment Testers


<h2> Introduction </h2>
<div>
This homework asks you to construct pctest which builds, and tests a given C program to assist an instructor of a C programming courses in evaluating programming homework submissions of the students. With each of given test inputs, p c test runs both a target program (i.e., a student’s submission) and a solution program (i.e., a correct program offered by the instructor), and then compares their results to determine whether the target program returns a correct result or a failure (i.e., wrong answer) or not.
 </div>
 

<h3> About Command Line <h3>

     $ pctest i [testdir] t [timeout] [solution] [target] 
  
A user gives <testdir> a path to a directory where the test input files are stored. All files under <testdir> will be recognized as input files. <timeout> specifies the time limit of a program execution in seconds. It should be an integer between 1 and 10. <solution> and <target> give a filename of the excutable file of the correct version and a filename of the excutable file of a student’s program under test, respectively.

 
 
<h2> Test Result </h2>
<div>
The summary include (1) the number of the correct test executions and the number of failed test executions, (2) the maximum and the minimum time of a single test execution, and the sum of all test execution time of target.
</div>

 <div>
  <h3> Checking Test Results <h3>
</div>
   
<div>
 pctest determines that target fails for a given test input if the corresponding test execution falls into one of the following cases:
  1) the program does not terminate within a certain amount of execution time (i.e., time over), or
  2) the text printed to the standard output is not identical to that of solution.
  3) the maximum execution time of a single test run in milliseconds, and
  4) the minimum execution time of a single test run in milliseconds, and
  5) the sum of the execution time of all test runs in milliseconds.
 </div>

<h2> Demo Link </h2>
https://youtu.be/zofdY-M8ZMU

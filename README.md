Simulator for [Efficient GPU hardware transactional memory through early conflict resolution](https://ieeexplore.ieee.org/abstract/document/7446071)

Build instructions for Ubuntu 22.04

1. Obtain GCC-4.8.5 (steps taken from [this post](https://askubuntu.com/questions/1450426/need-gcc-and-g-4-8-in-ubuntu-22-04-1)):
   ```
   mkdir /tmp/gcc-4.8 && cd /tmp/gcc-4.8
   wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-4.8/g++-4.8_4.8.   5-4ubuntu8_amd64.deb 
   wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-4.8/libstdc++-4.8-dev_4.   8.5-4ubuntu8_amd64.deb 
   wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-4.8/gcc-4.8-base_4.8.   5-4ubuntu8_amd64.deb 
   wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-4.8/gcc-4.8_4.8.   5-4ubuntu8_amd64.deb 
   wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-4.8/libgcc-4.8-dev_4.8.   5-4ubuntu8_amd64.deb 
   wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-4.8/cpp-4.8_4.8.   5-4ubuntu8_amd64.deb 
   wget http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-4.8/libasan0_4.8.   5-4ubuntu8_amd64.deb  
   sudo dpkg -i *.deb
   ```
2. Update alternatives so you can switch GCC versions more easily.
   ```
   sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9
   sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
   sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 4
   sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 4
   ```

   And then:
   1. Run `sudo update-alternatives --config gcc` and choose `gcc-4.8`
   2. Run `sudo update-alternatives --config g++` and choose `g++-4.8`

3. Obtain CUDA Toolkit 4.2 from [NVIDIA's archive site](https://developer.nvidia.com/cuda-toolkit-42-archive)
   And then, install it to `/usr/local/cuda-4.2`

4. Build this particular snapshot of GPGPU-Sim
   ```
   source BUILD_IT
   ```

---------------------------------------------

##Original GPGPU-Sim Readme

Welcome to GPGPU-Sim, a cycle-level simulator modeling contemporary graphics
processing units (GPUs) running GPU computing workloads written in CUDA or
OpenCL. Also included in GPGPU-Sim is a performance visualization tool called
AerialVision and a configurable and extensible energy model called GPUWattch.
GPGPU-Sim and GPUWattch have been rigorously validated with performance and
power measurements of real hardware GPUs. 

This version of GPGPU-Sim has been tested with CUDA version 2.3, 3.1 and 4.0.

Please see the copyright notice in the file COPYRIGHT distributed with this
release in the same directory as this file.

If you use GPGPU-Sim in your research, please cite:

Ali Bakhoda, George Yuan, Wilson W. L. Fung, Henry Wong, Tor M. Aamodt,
Analyzing CUDA Workloads Using a Detailed GPU Simulator, in IEEE International
Symposium on Performance Analysis of Systems and Software (ISPASS), Boston, MA,
April 19-21, 2009.

In addition, if you use the GPUWattch energy model in your research, please cite:

Jingwen Leng, Tayler Hetherington, Ahmed ElTantawy, Syed Gilani, Nam Sung Kim,
Tor M. Aamodt, Vijay Janapa Reddi, GPUWattch: Enabling Energy Optimizations in
GPGPUs, In proceedings of the ACM/IEEE International Symposium on Computer
Architecture (ISCA 2013), Tel-Aviv, Israel, June 23-27, 2013.

If you use figures plotted using AerialVision in your publications, please cite:

Aaron Ariel, Wilson W. L. Fung, Andrew Turner, Tor M. Aamodt, Visualizing
Complex Dynamics in Many-Core Accelerator Architectures, In Proceedings of the
IEEE International Symposium on Performance Analysis of Systems and Software
(ISPASS), pp. 164-174, White Plains, NY, March 28-30, 2010.

This file contains instructions on installing, building and running GPGPU-Sim.
Detailed documentation on what GPGPU-Sim models, how to configure it, and a
guide to the source code can be found here: <http://gpgpu-sim.org/manual/>.
Instructions for building doxygen source code documentation are included below.
Detailed documentation on GPUWattch including how to configure it and a guide
to the source code can be found here: <http://gpgpu-sim.org/gpuwattch/>.

If you have questions, please sign up for the google groups page (see
gpgpu-sim.org), but note that use of this simulator does not imply any level of
support.  Questions answered on a best effort basis.

To submit a bug report, go here: http://www.gpgpu-sim.org/bugs/

See Section 2 "INSTALLING, BUILDING and RUNNING GPGPU-Sim" below to get started.

See file CHANGES for updates in this and earlier versions.


1. CONTRIBUTIONS and HISTORY

== GPGPU-Sim ==

GPGPU-Sim was created by Tor Aamodt's research group at the University of
British Columbia.  Many have directly contributed to development of GPGPU-Sim
including: Tor Aamodt, Wilson W.L. Fung, Ali Bakhoda, George Yuan, Ivan Sham,
Henry Wong, Henry Tran, Andrew Turner, Aaron Ariel, Inderpret Singh, Tim
Rogers, Jimmy Kwa, Andrew Boktor, Ayub Gubran Tayler Hetherington and others.

GPGPU-Sim models the features of a modern graphics processor that are relevant
to non-graphics applications.  The first version of GPGPU-Sim was used in a
MICRO'07 paper and follow-on ACM TACO paper on dynamic warp formation. That
version of GPGPU-Sim used the SimpleScalar PISA instruction set for functional
simulation, and various configuration files indicating which loops should be
spawned as kernels on the GPU, along with reconvergence points required for
SIMT execution to provide a programming model simlar to CUDA/OpenCL.  Creating
benchmarks for the original GPGPU-Sim simulator was a very time consuming
process and the validity of code generation for CPU run on a GPU was questioned
by some.  These issues motivated the development an interface for directly
running CUDA applications to leverage the growing number of applications being
developed to use CUDA.  We subsequently added support for OpenCL and removed
all SimpleScalar code.

The interconnection network is simulated using the booksim simulator developed
by Bill Dally's research group at Stanford.

To produce output that matches the output from running the same CUDA program on
the GPU, we have implemented several PTX instructions using the CUDA Math
library (part of the CUDA toolkit). Code to interface with the CUDA Math
library is contained in cuda-math.h, which also includes several structures
derived from vector_types.h (one of the CUDA header files).

== GPUWattch Energy Model ==

GPUWattch (introduced in GPGPU-Sim 3.2.0) was developed by researchers at the
University of British Columbia, the University of Texas at Austin, and the
University of Wisconsin-Madison.  Contributors to GPUWattch include Tor
Aamodt's research group at the University of British Columbia: Tayler
Hetherington and Ahmed ElTantawy; Vijay Reddi's research group at the
University of Texas at Austin: Jingwen Leng; and Nam Sung Kim's research group
at the University of Wisconsin-Madison: Syed Gilani. 

GPUWattch leverages McPAT, which was developed by Sheng Li et al.  at the
University of Notre Dame, Hewlett-Packard Labs, Seoul National University, and
the University of California, San Diego. The paper can be found at
http://www.hpl.hp.com/research/mcpat/micro09.pdf. 



2. INSTALLING, BUILDING and RUNNING GPGPU-Sim

Assuming all dependencies required by GPGPU-Sim are installed on your system,
to build GPGPU-Sim all you need to do is add the following line to your
~/.bashrc file (assuming the CUDA Toolkit was installed in /usr/local/cuda):

	export CUDA_INSTALL_PATH=/usr/local/cuda

then type

	bash
	source setup_environment
	make

If the above fails, see "Step 1" and "Step 2" below. 

If the above worked, see "Step 3" below, which explains how to run a CUDA
benchmark on GPGPU-Sim.

Step 1: Dependencies
====================

GPGPU-Sim was developed on SUSE  Linux (this release was tested with SUSE
version 11.3) and has been used on several other Linux platforms (both 32-bit
and 64-bit systems).  In principle, GPGPU-Sim should work with any linux
distribution as long as the following software dependencies are satisfied.

Download and install the CUDA Toolkit. It is recommended to use version 3.1 for
normal PTX simulation and version 4.0 for cuobjdump support and/or to use
PTXPlus (Harware instruction set support). Note that it is possible to have
multiple versions of the CUDA toolkit installed on a single system -- just
install them in different directories and set your CUDA_INSTALL_PATH
environment variable to point to the version you want to use.

[Optional] If you want to run OpenCL on the simulator, download and install
NVIDIA's OpenCL driver from <http://developer.nvidia.com/opencl>.  Update your
PATH and LD_LIBRARY_PATH as indicated by the NVIDIA install scripts. Note that
you will need to use the lib64 directory if you are using a 64-bit machine.  We
have tested OpenCL on GPGPU-Sim using NVIDIA driver version 256.40
<http://developer.download.nvidia.com/compute/cuda/3_1/drivers/devdriver_3.1_linux_64_256.40.run>
This version of GPGPU-Sim has been updated to support more recent versions of
the NVIDIA drivers (tested on version 295.20). 

GPGPU-Sim dependencies:
* gcc
* g++
* make
* makedepend
* xutils
* bison
* flex
* zlib
* CUDA Toolkit
	
GPGPU-Sim documentation dependencies:
* doxygen
* graphvi

AerialVision dependencies:
* python-pmw
* python-ply
* python-numpy
* libpng12-dev
* python-matplotlib

We used gcc/g++ version 4.5.1, bison version 2.4.1, and flex version 2.5.35.

If you are using Ubuntu, the following commands will install all required
dependencies besides the CUDA Toolkit.

GPGPU-Sim dependencies:
"sudo apt-get install build-essential xutils-dev bison zlib1g-dev flex
libglu1-mesa-dev"

GPGPU-Sim documentation dependencies:
"sudo apt-get install doxygen graphviz"

AerialVision dependencies:
"sudo apt-get install python-pmw python-ply python-numpy libpng12-dev
python-matplotlib"

CUDA SDK dependencies:
"sudo apt-get install libxi-dev libxmu-dev libglut3-dev"

Finally, ensure CUDA_INSTALL_PATH is set to the location where you installed
the CUDA Toolkit (e.g., /usr/local/cuda) and that $CUDA_INSTALL_PATH/bin is in
your PATH.  You probably want to modify your .bashrc file to incude the
following (this assumes the CUDA Toolkit was installed in /usr/local/cuda):

	export CUDA_INSTALL_PATH=/usr/local/cuda
	export PATH=$CUDA_INSTALL_PATH/bin


Step 2: Build
=============

To build the simulator, you first need to configure how you want it to be
built. From the root directory of the simulator, type the following commands in
a bash shell (you can check you are using a bash shell by running the command
"echo $SHELL", which should print "/bin/bash"):

cd v3.x
source setup_environment <build_type>

replace <build_type> with debug or release. Use release if you need faster
simulation and debug if you need to run the simulator in gdb. If nothing is
specified, release will be used by default.  

Now you are ready to build the simulator, just run

make

After make is done, the simulator would be ready to use. To clean the build,
run

make clean

To build the doxygen generated documentations, run

make docs

to clean the docs run

make cleandocs

The documentation resides at v3.x/doc/doxygen/html.

Step 3: Run
============

Copy the contents of v3.x/configs/QuadroFX5800/ or v3.x/configs/GTX480/ to your
application's working directory.  These files configure the microarchitecture
models to resemble the respective GPGPU architectures.

To use ptxplus (native ISA) change the following options in the configuration
file to "1" (Note: you need CUDA version 4.0) as follows:

-gpgpu_ptx_use_cuobjdump 1
-gpgpu_ptx_convert_to_ptxplus 1

Now To run a CUDA application on the simulator, simply execute

source setup_environment <build_type>

Use the same <build_type> you used while building the simulator. Then just
launch the executable as you would if it was to run on the hardware. By
running "source setup_environment <build_type>" you change your LD_LIBRARY_PATH
to point to GPGPU-Sim's instead of CUDA or OpenCL runtime so that you do NOT
need to re-compile your application simply to run it on GPGPU-Sim.

To revert back to running on the hardware, remove GPGPU-Sim from your
LD_LIBRARY_PATH environment variable.

The following GPGPU-Sim configuration options are used to enable GPUWattch

	- power_simulation_enabled 1 (1=Enabled, 0=Not enabled)
	- gpuwattch_xml_file <filename>.xml 

The GPUWattch XML configuration file name is set to gpuwattch.xml by default and
currently only supplied for GTX480 (default=gpuwattch_gtx480.xml).  Please refer to
<http://gpgpu-sim.org/gpuwattch/> for more information. 

Running OpenCL applications is identical to running CUDA applications. However,
OpenCL applications need to communicate with the NVIDIA driver in order to
build OpenCL at runtime. GPGPU-Sim supports offloading this compilation to a
remote machine. The hostname of this machine can be specified using the
environment variable OPENCL_REMOTE_GPU_HOST. This variable should also be set
through the setup_environment script. If you are offloading to a remote machine,
you might want to setup passwordless ssh login to that machine in order to
avoid having too retype your password for every execution of an OpenCL
application.

If you need to run the set of applications in the NVIDIA CUDA SDK code
samples then you will need to download, install and build the SDK.

The CUDA applications from the ISPASS 2009 paper mentioned above are
distributed separately on the git server under the directory
ispass2009-benchmarks. The README.ISPASS-2009 file distributed with the
benchmarks now contains updated instructions for running the benchmarks on
GPGPU-Sim v3.x.


4.  (OPTIONAL) Updating GPGPU-Sim (ADVANCED USERS ONLY)

If you have made modifications to the simulator and wish to incorporate new
features/bugfixes from subsequent releases the following instructions may help.
They are meant only as a starting point and only recommended for users
comfortable with using source control who have experience modifying and
debugging GPGPU-Sim.

WARNING: Before following the procedure below, back up your modifications to
GPGPU-Sim. The following procedure may cause you to lose all your changes.  In
general, merging code changes can require manual intervention and even in the
case where a merge proceeds automatically it may introduce errors.  If many
edits have been made the merge process can be a painful manual process.  Hence,
you will almost certainly want to have a copy of your code as it existed before
you followed the procedure below in case you need to start over again.  You
will need to consult the documentation for git in addition to these
instructions in the case of any complications.

STOP.  BACK UP YOUR CHANGES BEFORE PROCEEDING. YOU HAVE BEEN WARNED. TWICE.

To update GPGPU-Sim you need git to be installed on your system.  Below we
assume that you ran the following command to get the source code of GPGPU-Sim:

git clone git://dev.ece.ubc.ca/gpgpu-sim

Since running the above command you have made local changes and we have
published changes to GPGPU-Sim on the above git server. You have looked at the
changes we made, looking at both the new CHANGES file and probably even the
source code differences.  You decide you want to incorporate our changes into
your modified version of GPGPU-Sim.  

Before updating your source code, we recommend you remove any object files:

make clean

Then, run the following command in the root directory of GPGPU-Sim:

git pull

While git is pulling the latest changes, conflicts might arise due to changes
that you made that conflict with the latest updates. In this case, you need to
resolved those conflicts manually. You can either edit the conflicting files
directly using your favorite text editor, or you can use the following command
to open a graphical merge tool to do the merge:

git mergetool

Now you should test that the merged version "works".  This means following the
steps for building GPGPU-Sim in the *new* README file (not this version) since
they may have changed. Assuming the code compiles without errors/warnings the
next step is to do some regression testing.  At UBC we have an extensive set of
regression tests we run against our internal development branch when we make
changes.  In the future we may make this set of regression tests publically
available. For now, you will want to compile the merged code and re-run all of
the applications you care about (implying these applications worked for you
before you did the merge). You want to do this before making further changes to
identify any compile time or runtime errors that occur due to the code merging
process. 




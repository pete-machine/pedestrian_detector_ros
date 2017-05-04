## Install scikit-image

    pip install scikit-image
    
## Setting up MATIO (matlab input/output interface)
Download it: http://sourceforge.net/projects/matio/
Execute the following commands in the directory of the downloaded file.
	
	tar zxf matio-X.Y.Z.tar.gz
    cd matio-X.Y.Z
    ./configure (you might need to make this file an executable)
    make
    make check
    sudo make install

Check the src/pedestriandetectorros/CMakeLists.txt to make it point to the right position of libmatio.so

### Error "undefined reference to H5"
If you get an error like: "undefined reference to H5...".:
Delete the installation folder matio-X.Y.Z by "rm -R matio-X.Y.Z"
	
	tar zxf matio-X.Y.Z.tar.gz
    cd matio-X.Y.Z
    chmod +x ./configure
    ./configure --without-hdf5
    make
	sudo make install

## Test pedestrian package using web-cam
Git clone [usb_cam](https://github.com/bosch-ros-pkg/usb_cam) into workspace.
    
    git clone https://github.com/bosch-ros-pkg/usb_cam 
    
Run launch file.

    roslaunch pedestrian_detector localTestPed.launch

## Test pedestrian package on bag
Run bag and launch file.

    roslaunch pedestrian_detector onBag.launch
    

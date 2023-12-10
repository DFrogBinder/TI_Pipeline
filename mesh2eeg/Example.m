%% Paolo Giacometti
%
% This is a script that shows an example of how to use the Mesh2EEG
% package. The main function to run is ComputeEEGPos.m, which calls the other
% functions in the package. The others can be used independently for
% specific cases. 
%
% In this example, a simple sphere is used to simulate a head. The boundary
% element mesh of a sphere with radius 20 and centered at the origin is
% included in ExampleSphere. 

%% Load boundary element (triangular) mesh
load ExampleSphere.mat

%% Specify fiducial position coordinates on mesh

% Nasion position (indentation at the top of the nose approximately between
% the eyebrows)
Nz = [0 20 0]; 
% Inion position (indentation at the back of the head approximately where
% the neck begins)
Iz = [0 -20 0]; 
% Left (as seen from above with nose on top) preauricular position 
% (indentation in front of the top of the ear cannal, dent between the 
% upper edge of the targus and the daith)
M1 = [-20 0 0];
% Right (as seen from above with nose on top) preauricular position 
% (indentation in front of the top of the ear cannal, dent between the 
% upper edge of the targus and the daith)
M2 = [20 0 0];

% Compile fiducial position matrix for input into function
Fiducials = [Nz;Iz;M1;M2];

% Run function. 
% Inputs: Fiducial positions
%         Mesh Faces (surface elements)
%         Mesh Vertices (nodes)
%         Layout option (1: 10-5, 2: 10-10, 3: 10-5)
%         Coordinate transformation option (0: leaves the coordinate system
%         as is, 1: transforms the coordinates of the mesh vertices to
%         align coordinate system with fiducials, such that the origin
%         lies in the midpoint between M1 and M2 and Nz lies normal to the
%         segment M1-M2 in the negative x-direction.
% Outputs: [x,y,z] coordinates of each EEG position
%          Labels for each position 
[EEGPts1,EEGLab1] = ComputeEEGPos(Fiducials,Sphere.Faces,Sphere.Vertices,1,0);
[EEGPts2,EEGLab2] = ComputeEEGPos(Fiducials,Sphere.Faces,Sphere.Vertices,2,0);
[EEGPts3,EEGLab3] = ComputeEEGPos(Fiducials,Sphere.Faces,Sphere.Vertices,3,0);


figure
patch(Sphere,'FaceColor',       [0.8 0.8 1.0], ...
         'EdgeColor',       'none',        ...
         'FaceLighting',    'gouraud',     ...
         'AmbientStrength', 0.15);

% Lights, axis, and view
camlight('headlight');
material('dull');
% light('Position',[0 -.75 -0.5],'Style','infinite'); % add extra lights
axis('image');
view([-135 35]);


% Visualize points
hold on
plot3(EEGPts1(:,1),EEGPts1(:,2),EEGPts1(:,3),'.','MarkerSize',15)
plot3(EEGPts2(:,1),EEGPts2(:,2),EEGPts2(:,3),'ro','MarkerSize',10,'LineWidth',2)
plot3(EEGPts3(:,1),EEGPts3(:,2),EEGPts3(:,3),'yh','MarkerSize',10,'LineWidth',2)

% Create legend
legend('Mesh','EEG 10-5 positions','EEG 10-10 positions','EEG 10-20 positions')

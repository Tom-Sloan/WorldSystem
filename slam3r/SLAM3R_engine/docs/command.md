You are an expert programmer and excel at debugging issues.
Do not leave TODOs or placeholders.

Carefully and extremly deeply. Read through the history sections of this. Make sure to carefully review all the files. The documents of the SLAM3D and mesh on your service containers just closed and are no longer running. You will see that when SLAM3D sends the keyframes to the mesh services, they are empty.

<HISTORY>
I previously had slinger_fusing.py working and it would send the points, 3D points to get downsampled and then the mesh. This was very slow so I abstracted the method and created streaming. I'm using streaming.py and want the 3D shared memory to generate a mesh that we send to rerun.
<HISTORY>

<OUTOUT>
slam3r_processing should output 3d points for the mesh_service to use. Come up with three different possible solutions to solve this issue. And present the best one.
<OUTPUT>

<FILES>
Files to review:
/home/sam3/Desktop/Toms_Workspace/WorldSystem/slam3r/SLAM3R_engine/app.py
/home/sam3/Desktop/Toms_Workspace/WorldSystem/slam3r/SLAM3R_engine/recon.py
The slam3r paper is at https://arxiv.org/html/2412.09401v3
For a working commit of slam3r_processing.py look at https://github.com/Tom-Sloan/WorldSystem/commit/b15afedda8b36cb8423df86b0f4b9e72a23d6b9b
/home/sam3/Desktop/Toms_Workspace/WorldSystem/docker-compose.yml
https://github.com/PKU-VCL-3DV/SLAM3R/tree/main
<FILES>

<NOTES>
app.py and recon.py are a working demo from the original slam3r repo.
Use conda 3dreconstruction for local developement (at /home/sam3/anaconda3/envs/3dreconstruction/bin/python), if there are an libraries missing from 3drecontstruction, install them in 3dreconstruction.
Don't build docker containers, ask me to build them.
<NOTES>

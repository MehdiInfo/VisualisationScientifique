# VisualisationScientifique

## Travail fait :

Implémentation de différentes version de Ray Tracing pour le rendu de volum et de maillage en utilisant MIP, MEAN, LMIP, MINIP et DEMIP, ces derniers peuvent être retrouvé au niveau du dossier shader dans le ficheir volume.cu.

Une modification au niveau de l'interface graphique de l'application a était faite aussi dans le fichier screenDisplay.cpp pour avoir un rendu spécifique en fonction du type de ray tracing choisie.

# Compiler le dépôt

1. Cloner le dépôt
`git clone --recursive https://romeogit.univ-reims.fr/mnoizet/VisualisationScientifique.git`

2. Dans le répertoire du dépôt, créez un sous-répertoire build
```
mkdir build
cd build
```

3. Lancer la commande
` cmake -DOptiX_INSTALL_DIR=/usr/include/OptiX/ ..` //Pour l'instant il y a des warning concernant le policy, ils seront bientôt corrigés

```
4. Compiler les projets
`cmake --build . --config Debug `
ou 
`make -j 4`

# Dépendances

- CUDA : Api de calcul sur GPU
>> Pour télécharger cuda, il faut se rendre sur le site https://developer.nvidia.com/cuda-downloads .

- OptiX : API de Ray tracing
>> Pour télécharger OptiX, il faut se rendre sur le site https://developer.nvidia.com/designworks/optix/download. 
La version proposée est la 7.4.0, cependant elle nécessite la version NVIDIA R495.89 ou plus récentes des drivers graphiques.
D'autres version d'OptiX sont disponnibles sur ce lien : https://developer.nvidia.com/designworks/optix/downloads/legacy .
Voici une petite liste de la version OptiX et la version minimal du driver requis : 

| Optix version      | Driver version   |
| -----------        | -----------      |
| OptiX 7.4.0        | NVIDIA R465.84   |
| OptiX 7.3.0        | NVIDIA R456.71   |
| OptiX 7.1.0        | NVIDIA R450      |
| OptiX 7.0.0        | NVIDIA R435.80   |

- ImGUI : Interfaçage
> Le lien du git : https://github.com/ocornut/imgui
- GLFW : création du contexte graphique sous OpenGL
> Le lien du site officiel : https://www.glfw.org/
- GLEW : extension OpenGL
> Le lien du site officiel : http://glew.sourceforge.net/
- STB : I/O image file format
> Le lien du github : https://github.com/nothings/stb
- ASSIMP : I/O 3D file format
> Le lien du site officiel : https://www.assimp.org/

# Runtime

- Lancer `1_meshVisualization` pour exécuter le programme. Il n'y a pas de paramètre à passer.
 Par défaut, le maillage statue.obj et le volume cafard.dat seront pris en charge.
 Exécution de 1_meshVisualization : 

-![Mesh visualisation](/images/sample1.PNG)

Exécution de 2_volumeVisualization : 

-![Mesh visualisation](/images/sample2.PNG)

Exécution de VSProject : 

-![Mesh visualisation](/images/sample3.PNG)


# Installing Raster Tools

To install raster tools we suggest using a package manager like
[Anaconda](https://www.anaconda.com/products/distribution) or
[Minconda](https://docs.conda.io/en/latest/miniconda.html).
In this example we demonstrate how to install Raster-Tools and its dependecies
using Miniconda.


## Miniconda GUI

Steps:

1. Download Minconda for your os (I will be using Windows)
2. Launch the installer and select "Just Me"

    ![Screenshot of Miniconda installer](https://user-images.githubusercontent.com/11561085/200663300-db46d3e7-3787-4d9a-adbe-3774b33dfa33.png)

3. Launch anaconda powershell prompt:

    ![Screenshot of PowerShell prompt in Windows Start menu](https://user-images.githubusercontent.com/11561085/200663596-be258e6b-64bc-48f7-9e04-92ec6474f6ce.png)

4. Download our yml file <a href="./rstools39.yml" download>Click to Download</a> and store it within your documents folder in a location you can access (e.g., ./documents/raster_tools_setup/rstools39.yml
5. Type in `conda create -f <the path to the yml file>`. For example: `conda create -f ./documents/raster_tools_setup/rstools39.yml`
6. Press Enter and Raster-Tools will be installed within the rstools39 environment

To launch Jupyter Lab type the following commands:

```powershell
conda activate rstools39
jupyter lab
```

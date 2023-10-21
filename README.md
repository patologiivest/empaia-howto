# Empaia API "How-to?"

This repository exemplifies how to develop custom algorithms (plugins) for the [Empaia Platform](https://empaia.org).
For a much more in-depth documentation, I will refer to their [official documentation](https://developer.empaia.org/#/).
The general workflow when you plan to develop your own algorithm for their market place is as follows:

1. Prepare your algorithm 
1. Write an _Emapaia App Description (EAD)_
1. Set up testing environment locally (optional)
1. Understand the Empaia REST API (optional)
1. Write glue code 
1. Containerize your application

## Step 1: Prepare your application 

We assume that you have your image analysis machine learning algorithm ready, which can be run _headlessly_.
In this guide, we only consider programming language since it is the most popular among machine learning afficinados. 
If you want to use another language, in the final step "Containerization", you have to adjust your `Dockerfile` 
accordingly. If you do not know how to do this, ask [past@hvl.no](mailto:past@hvl.no).

Hence, as a first step you should clarify for yourself how the following architecture will look concretely for your 
algorithm, i.e.:

- What user inputs aside from the _whole slide image (WSI)_ are needed?
- What outputs should your algorithm produce: a single statistic (floating point number), structured information, or 
annotations (e.g. when segmenting),
- What other dependencies your algorithm (external binaries, local config files, neural network weights from a training
phase)?

For the remainder, we assume that your algorithm lives in a file called `alg.py` and is encapsulated as a function:
```python
def run_algorithm(tiles, other_input_arg, ...):
   ...

```
where `tiles` is expected to be an iterable containing relevant image tiles of the whole slide image, 
`other_input_arg` and following are other user input parameters (depending on you algorithm).
Also, all required Python packages and their version numbers should be explicated within an `Requirements.txt` file.
The latter can easily be creted by first creating a virtual Python environment, installing all necessary dependencies
in it, and then running `pip freeze > Requirements.txt` (or you may decide to use [Poetry](https://python-poetry.org/)
instead :wink:).

## Step 2: Write an "EAD"

Each Empaia "App" on the marketplace consists of two things:

- an EAD description, 
- a zipped container image of the application.

The former is a simple JSON file that contains _meta-information_ abut the algorithm, i.e. inforation that should 
be presented to the user in the user interface and, most importantly, the in- and outputs of the application. 

Below, you find an example of how such a `ead.json` file may look like:
```json
{
    "$schema": "https://gitlab.com/empaia/integration/definitions/-/raw/main/ead/ead-schema.v3.json",
    "name": "My Cool Medical AI Algorithm",
    "name_short": "Cool App",
    "namespace": "org.empaia.helse-vest-piv.cool_app.v3.1",
    "description": "Does super advanced AI stuff, you know...",
    "io": {
        "my_wsi": {
            "type": "wsi"
        },
        "my_quantification_result": {
            "type": "float",
            "description": "Human readable text, e.g. super important metric",
            "reference": "io.my_wsi"
        }
    },
    "modes": {
        "standalone": {
            "inputs": [
                "my_wsi"
            ],
            "outputs": [
                "my_quantification_result"
            ]
        }
    }
}
```
The fields
- `$schema`,
- `name`,
- `name_short`,
- `namespace`,
- `description`,
- `io`, and 
- `modes`

are mandatory! The `$schema` property is fixed and links to [JSON schema definition](https://gitlab.com/empaia/integration/definitions/-/blob/main/ead/ead-schema.v3.json).
The properties `name`, `name_short`, and `description` are used to display a user readable (short) title and 
description in the Empaia UI. The `namespace` servces as a _unique_ identifier for you app. Therefore, it has to 
follow a specific naming scheme, see [here](https://developer.empaia.org/app_developer_docs/v3/#/specs/ead?id=namespace)!
The `io` section defines all the parameters that will be used as in- or output for the app, see respective section below.
Finally, the `modes` section defines the _execution mode_ in which the app will run and what in- and output parameters 
will be used then. For our purposes, we will stick with the `standalone` mode most of the time.

Have a detailed look [here](https://developer.empaia.org/app_developer_docs/v3/#/specs/ead) if you want to see the complete spec!

## In- and Output parameters

Parameters are specified inside the `io` section of the `ead` and follow a fixed structure:
```json
{
    "<name>" : {
        "type": "<see below>",
        "description": "<optional, human-readable presentation>",
        "reference": "<optional, reference to another parameter>"
    }
}
```
The Empaia supported parameter types can be classified into the following categories

- the whole slide image itself (`"type": "wsi"`), there will be always **exactly one** such parameter, you can only customize its name and description,  
- primitive values, i.e. `"type": "integer|float|bool|string"`, can be used for scalar values, which can be integer/floating point numbers, truth values, or character sequences (text),
- graphical annotations where `"type"` takes one of `point`, `line`, `arrow`, `rectangle`, `polygon`, `circle`,
- collections (`"type": "collection"`) wrap another parameter type to allow for multiple occurances of the wrapped type, e.g. sequence of rectangles,
- finally there is also a `class` parameter type, which represents catgorical values. In general, this type is used in combination
with annotations and also requires to define the classes at the root of the `ead` (see [details](https://developer.empaia.org/app_developer_docs/v3/#/specs/ead?id=class-data-type)).

The `reference` property plays an important role for assembling a structured result:
In order for the generic Emapai App UI to display the results of your algorithm, they all have to link back to the `wsi`
parameter! The general idea is that references should induce a tree structure with the `wsi` parameter on top, see [also](https://developer.empaia.org/app_developer_docs/v3/#/specs/references?id=structured-presentation):
```
my_wsi
├── my_quantification_result
└── my_segmentation_results
    ├── recatangle_0
        └── class_good
    ├── recatangle_1
        └── class_bad
    ...
    └── recatangle_1
        └── class_good
```

## Set up testing environment: Empaia App Test Suite (EATS)

Empaia provides a test suite such that you can test your application before handing it over to Empaia to make sure 
it does what it is expected to do. 

The eats test suite spuns up a whole infrastructure of services on your machine: Think a simplified version of the
Empaia platform. To this end, the container engine [Docker](https://www.docker.com/) is used (and this means specifically Docker! 
Alternative container engines such as Podman, containerd, LXC do not work...).
Hence, you have docker installed on your system first.

> [!NOTE]
> The container virtualization technology underlying Docker and co. is provided by powers of the Linux kernel.
> Hence, containers basically _only_ work under Linux. However, with some "tricks" containers can be also run on 
> Windows and Mac OS X (using a hypervisor). This is also the reason why Docker is _free to use_ only on Linux 
> and for private or open source projects only when you are on Windows or Mac. 
> If you plan to use Docker for Windows/Mac in an organizational context, you have to obtain a licence!
> Also the performance of containers on Windows/Mac will always be sub-par compared with their Linux equivalent
> and should therefore only be used for testing purposes.

Follow the official guide on how to install Docker for your operating system:
- [Mac Install Guide](https://docs.docker.com/desktop/install/mac-install/)
- [Windows Install Guide](https://docs.docker.com/desktop/install/windows-install/)
- [Linux Install Guide](https://docs.docker.com/desktop/install/linux-install/)

When Docker is installed you can install `eats` simply as a [python package](https://pypi.org/project/empaia-app-test-suite/), 
i.e. open a up a command line (if you use windows, you have to enter a Linux shell in WSL via the Windows Terminal) and type:
```bash
pip install empaia-app-test-suite
```
or if the `pip` binary should not be available on your `$PATH` by accident:
```
python3 -m pip install empaia-app-test-suite
```

Next, you should collect some slide images that you want to use in the test suite and place them into a working directory.
Navigate to a nice location in you file system using the command line (recall that you use `cd` to change directories and 
`pwd` to see where you currently are), create a folder called `eats`, including a sub-folder
for the `images`:

```bash
mkdir eats
cd eats
mkdir images
```

Open a file explorer at the respective location and then move the slide images into the `images` folder.

Before, we can start EATS, we have to create one more json file called `wsi-mount-points.json` and place it directly
beneath the `eats` directory. The content of this file will look something like:
```json
{
    "/global/path/to/eats/images": "/data"
}
```
the `/global/path/to/` is a placeholder for the fully-qualified path from the root of you file system to the newly
created `eats` directory. 

> [!NOTE]
> As a windows user you will necissarily run EATS from within _Windows Subsystem for Linux (WSL)_ in order to run Docker.
> Thus, paths in the Windows file system have to be aligned, concretely:
> `C:\Users\<username>\Downloads\eats\`
> becomes
> `/mnt/c/Users/<username>/Downloads/eats/`
> in WSL.

Finally, you are able to start EATS with the command:
```bash
eats services up ./wsi-mount-points.json
```

On the first startup, this may take a while since it will download the required container images onto your machine 
and start them.
When everything started, you can access the Empaia workbench under the following URL 
```
http://localhost:8888/wbc3/
```
However, you won't see any cases nor slides.
You first have to register them manually. 

Create another json file with following content: 
```json
{
    "type": "wsi",
    "path": "/data/<image-name>",
    "id: "<id>"
}
```
The `<image-name>` must coincide with the file name of an image in the `/eats/images` folder. 
The `<id>` can be arbitrarily chosen.
Name the file something like `slide1.json`.
Now you can register the slide with:
```bash
eats slides register slide1.json
```

You have to repeat this process for all your images in order to see them in the workbench client. 
Also make sure that the slides receive distinct ids!




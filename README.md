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

For the remainder, we assume that your algorithm lives in a file called `alg.py` and is defined as a function:
```python
def run_algorithm(tiles, potential_order_input, ...):
   ...

```
where `tiles` is expected to be an iterable containing relevant image tiles of the whole slide image, 
`potential_order_input` and following are other user input parameters (depending on you algorithm).
Also, all required Python packages and their version numbers should be explicated within an `Requirements.txt` file.
The latter can easily be creted by first creating a virtual Python environment, installing all necessary dependencies
in it, and then running `pip freeze > Requirement.txt` (or you may decide to use [Poetry](https://python-poetry.org/)
instead :wink:).

## Step 2: Write "EAD"

Each Empaia "App" on the marketplace consists of two things:

- an EAD description, 
- a zipped container image of the application.

The former is a simple JSON file that contains _meta-information_ abut the algorithm, i.e. inforation that should 
be presented to the user in the user interface and, most importantly, the in- and outputs of the application. 

Below, you find an example of how such a `ead.json` file may look like:
```
{
    "$schema": "https://gitlab.com/empaia/integration/definitions/-/raw/main/ead/ead-schema.v3.json",
    "name": "My Cool Medical AI Algorithm",
    "name_short": "myAIalgv.1.0",
    "namespace": "no.helse-bergen.piv",
    "description": "Does super advanced AI stuff, you know...",
    "io": {
        "my_wsi": {
            "type": "wsi"
        },
        "my_quantification_result": {
            "type": "float"
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

Have a look [here](https://developer.empaia.org/app_developer_docs/v3/#/specs/ead) to see the complete spec!



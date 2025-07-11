# Firecrown: (DES year 1); 6x2pt analysis

Here we take a trial attempt at the [firecrown](https://firecrown.readthedocs.io/en/latest/_static/intro_article.html)
& [cosmoSIS](https://cosmosis.readthedocs.io/en/latest/index.html) interplay.
Using the structure taken from the examples and tutorials of firecrown, cosmoSIS and [SACC](https://sacc.readthedocs.io/en/latest/intro.html); 
we build a structure that allows us to build a 6x2pt likelihood.

Note: for building the galaxy redshift distributions, we use the [glass.observations module](https://glass.readthedocs.io/v2023.6/reference/observations.html#rd48878c677a1-1)

To start, we might use the DES year 1 data since this is used and shown in the Firecrown 
tutorials & examples. However, we aim to build the structure in the general case, where we 
can run a 6x2pt forecast.

The elements that we need are:
- A SACC file generator;
    In the SACC file, we provide firecrown with the essential information to build a likelihood
- A SACC file, as a result of the SACC file generator 
- A likelihood function.
    Here we define the firecrown TwoPoint objects that allows us to define the likelihood.
    This will draw information from the SACC file (the systematics, etc.)
- The cosmoSIS 'ini' file
    Passing the essential info to cosmoSIS:
        - The pipeline and the modules required for this pipeline
        - The firecrown likelihood
        - The Sampler
        - The values and priors used for the sampling 

- The values and priors for the sampling.

The aim and contents of each of the files is more explicitly explained within these files.
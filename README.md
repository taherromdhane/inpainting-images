# inpainting-images

Implementation of an algorithm to inpaint images, along with a demo site to showcase the project. 
Image inpainting is the process of filling a region of the image that has been masked, such as it is seamless after the inpainting as if nothing was removed.
This has many use-cases in image processing, especially in removing unwanted objects from images (which is the case we focused on here).

You can try the demo [here](https://the-inpainter.herokuapp.com/).
You can also find the paper [here](https://www.irisa.fr/vista/Papers/2004_ip_criminisi.pdf).

## The Algorithm in a Nutshell

After initialization, the inpainter looks for the border pixels (pixels that are part of the image and there is a masked pixel in their immediate neighborhood).
It then iterates until there are no border pixels left (masked region is fully filled).
In each iteration, it does :

- It finds the border pixel with the most priority (according to the formula in the paper).
- Then, it tries to fill the patch which center is this pixel. 
- For this, it performs a search for the best patch to use in the filling among patches that are fully in the image.
- After filling, it updates the mask, the confidence values, and the border pixels.
    
This is showcased in this code :
```python

target_pixel, Cp = self._getMaxPriority()

opt_patch = self._getOptimalPatch(target_pixel)

self._updateConfidence(Cp, target_pixel)

self._fillPatch(target_pixel, opt_patch)

self._updateBorder(target_pixel)

```

You can check the code and the documentation to get a better understanding of how it works.


## Running the Demo

The demo is ready to be run, as-is, in your local environment. You can run it with python or with docker. 

### Running With Python

1. Install the dependencies

From the project folder, run this command
```shell
pip install -r requirements.txt
```

2. Run the main file

Execute this command in a shell or cmd to run the server from the 'app.py' file.
```shell
python app.py
```
Now you can check out the demo at 

### Running With Docker

1. Build image

Make sure that docker is installed, then from the cmd or shell run this command
```shell
docker build -t inpainter:1.0 .
```

make sure to use this line 
```docker
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:server", "--timeout", "600"]
```
instead of thie line 
```docker
CMD ["gunicorn", "app:server", "--timeout", "600"]
```

to fire the server at the right port so we can access it later on. This is not done in deployment that's why it's commented. 

2. Run image

After building image, you just need to run a container with it. Run this command
```shell
docker run -p 8000:8000 inpainter:1.0
```

## Deploying

This project is ready to be deployed with docker, you just need to comment this line
```python
# To run server in local environment
app.run_server(debug=True, dev_tools_hot_reload = False)
```

and uncomment these lines
```python
# To run server when deployed
# http_server = WSGIServer(('0.0.0.0', int(os.environ.get("PORT", 5000))), server)
# print(int(os.environ.get("PORT", 5000)), file=sys.stderr)
# http_server.serve_forever()
```

in the app.py file, starting at line 380.

## Documentation

You can find the full documentation of this project here [Docs](https://taherromdhane.github.io/inpainting-images/)

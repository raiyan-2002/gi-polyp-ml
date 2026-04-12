# gi-polyp-ml

download all the folders from this google drive [link](https://drive.google.com/drive/folders/1Bwbz3EF7SkfZXx2IWcrFn4K9Q35p75H4)

to create the environment, run the following command:

```
python -m venv venv

source venv/bin/activate  # on Windows, use `venv\Scripts\activate`

pip install -r requirements.txt
```


to test the models, you can run the following code:

```
python compare_models_random_image.py --output-path ./tests/testx.png
```

this will run the code on a random image from the test set and save the output to `./tests/testx.png`. You can change the output path as needed.

the images and masks are from: [here](https://datasets.simula.no/kvasir-seg/) (kvasir-seg.zip)
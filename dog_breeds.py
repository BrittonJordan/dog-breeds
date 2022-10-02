from fastai.vision.all import *
from fastcore.all import *
from duckduckgo_search import ddg_images

from typing import List
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"


def search_images(term, max_images=200): return L(ddg_images(term, max_results=max_images)).itemgot('image')


def download_imagaes_from_terms(terms: List, path_to_parent_dir: str, max_images = 2):
    for search in terms:
        dest = (path_to_parent_dir/search)
        dest.mkdir(exist_ok=True, parents=True)

        urls = search_images(search, max_images=max_images)
        download_images(dest, urls=urls, preserve_filename=True)
        time.sleep(10)

        resize_images(dest, max_size=400, dest=dest)

if __name__ == '__main__':
    PATH = Path("/Users/brittonjordan/Documents/personal-repos/dog-breeds/images")

    # download_imagaes_from_terms(searches, path, max_images=200)

    # failed = verify_images(get_image_files(path))
    # failed.map(Path.unlink)
    # len(failed)

    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(PATH)

    dls.show_batch(max_n=6)

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(3)

    category,_,probs = learn.predict(PILImage.create('/Users/brittonjordan/Documents/personal-repos/dog-breeds/Newfoundland1.jpeg'))
    print(f"This is a: {category}.")
    print(f"Probability it's a {category}: {probs[0]:.4f}")

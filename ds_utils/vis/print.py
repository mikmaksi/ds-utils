from typing import Union

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import Figure
from plotnine import save_as_pdf_pages
from plotnine.ggplot import ggplot
from seaborn.matrix import ClusterGrid


def save_multi_page_pdf(plist: list[Union[Figure, ggplot]], file: str):
    if isinstance(plist[0], ggplot):
        save_as_pdf_pages(plist, file)
    else:
        with PdfPages(file) as pdf:
            for p in plist:
                if isinstance(p, ClusterGrid):
                    p = p.fig
                pdf.savefig(p)

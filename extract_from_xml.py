import pmc_python
from lxml import etree

test_file = 'data/PMC2231364.xml'
with open(test_file, 'rb') as f:
    doc = etree.fromstring(f.read())

markdown_output = pmc_python.to_markdown(doc)

with open('sample_output.md', 'w+', encoding='utf-8') as f:
    f.write(markdown_output)

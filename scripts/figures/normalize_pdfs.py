from pathlib import Path
from pypdf import PdfReader,PdfWriter
P=Path(__file__).resolve().parent/'output'
for n in (3,4):
    p=P/f'Figure{n}.pdf';reader=PdfReader(p);writer=PdfWriter()
    assert len(reader.pages)==1
    original=reader.pages[0].get_contents().get_data()
    writer.add_page(reader.pages[0]);writer.add_metadata({'/Creator':'R 4.5.3 / ggplot2 4.0.3','/Title':f'Figure {n}'})
    tmp=p.with_suffix('.normalized.pdf');writer.write(tmp)
    check=PdfReader(tmp,strict=True);assert check.pages[0].get_contents().get_data()==original
    tmp.replace(p)
print('Normalized PDF object tables; page drawing commands preserved.')

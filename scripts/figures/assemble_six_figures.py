from pathlib import Path
import sys,fitz
here=Path(__file__).resolve().parent
folder=Path(sys.argv[1]) if len(sys.argv)>1 else here/'figures/six_main/pdf'
doc=fitz.open()
for n in range(1,7):
 with fitz.open(folder/f'Figure{n}.pdf') as one:doc.insert_pdf(one)
tmp=folder/'Main_Figures.pending.pdf';doc.save(tmp,garbage=4,deflate=True);tmp.replace(folder/'Main_Figures.pdf')
print('Assembled six current figure masters')

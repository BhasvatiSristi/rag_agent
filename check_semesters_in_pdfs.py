"""Check what semesters actually exist in the PDFs."""
import pdfplumber
from pathlib import Path
import re

data_dir = Path('data/raw')
for pdf_file in sorted(data_dir.glob('*.pdf')):
    print(f'\n{"="*60}')
    print(f'{pdf_file.name}')
    print("="*60)
    
    with pdfplumber.open(str(pdf_file)) as pdf:
        full_text = ''
        for page in pdf.pages:
            full_text += page.extract_text() + '\n'
        
        # Find all semester mentions
        semesters = sorted(set(re.findall(r'Semester (\d)', full_text)))
        print(f'Semesters found: {semesters}')
        
        # Check if Semester 3 appears
        if 'Semester 3' in full_text:
            print('✓ Semester 3 FOUND')
            idx = full_text.find('Semester 3')
            context = full_text[max(0, idx-100):idx+300]
            print(context)
            print("...")
        else:
            print('✗ Semester 3 NOT FOUND')

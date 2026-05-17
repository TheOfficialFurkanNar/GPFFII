import PyPDF2
import sys

def pdf_to_text(pdf_file_path, output_txt_path):
    # Open the PDF file in read-binary mode
    with open(pdf_file_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""

        # Loop through all pages and extract text
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"

    # Save extracted text to a file
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

    print(f"Text extracted and saved to {output_txt_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pdf_convert.py <pdf_file_path> <output_txt_path>")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    output_file = sys.argv[2]
    pdf_to_text(pdf_file, output_file)

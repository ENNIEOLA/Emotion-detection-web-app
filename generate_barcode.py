
import barcode
from barcode.writer import ImageWriter

data = "123456789012"

#)
Code128 = barcode.get_barcode_class('code128')
barcode_obj = Code128(data, writer=ImageWriter())

options = {
    "module_width": 0.2,     
    "module_height": 15,     
    "font_size": 10,         
    "text_distance": 2,     
    "quiet_zone": 1,        
}

filename = barcode_obj.save("my_barcode", options)

print(f"✅ Barcode generated and saved as {filename}.png")

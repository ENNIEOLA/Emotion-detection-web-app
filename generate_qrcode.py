import qrcode

data = "https://instagram.com/shop_elyra"

qr = qrcode.QRCode(
    version=1,  
    error_correction=qrcode.constants.ERROR_CORRECT_L, 
    box_size=10,  
    border=4,
    )


qr.add_data(data)
qr.make(fit=True)
img = qr.make_image(fill_color="black", back_color="white")
img.save("my_qrcode.png")
print("QR code generated and saved as my_qrcode.png!")


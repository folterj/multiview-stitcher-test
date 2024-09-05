

def int_to_rgba(intrgba: int) -> list:
    signed = (intrgba < 0)
    rgba = [x / 255 for x in intrgba.to_bytes(4, signed=signed, byteorder="big")]
    if rgba[-1] == 0:
        rgba[-1] = 1
    return rgba


def rgba_to_int(rgba: list) -> int:
    intrgba = int.from_bytes([int(x * 255) for x in rgba], signed=True, byteorder="big")
    return intrgba


def rgba_to_hexrgb(rgba: list) -> str:
    hexrgb = ''.join([hex(int(x * 255))[2:].upper().zfill(2) for x in rgba[:3]])
    return hexrgb


def hexrgb_to_rgba(hexrgb: str) -> list:
    rgba = int_to_rgba(eval('0x' + hexrgb + 'FF'))
    return rgba

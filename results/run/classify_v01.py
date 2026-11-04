"""Analytic colour, texture and geometry evidence; all constants chosen by visual meaning."""
import numpy as np
from PIL import Image
import cv2

CLASSES = ('tench','english_springer','cassette_player','chain_saw','church','french_horn','garbage_truck','gas_pump','golf_ball','parachute')


def features(rgb):
    # A 160-pixel square offers enough edges for object structure at a small fixed cost.
    a = cv2.resize(rgb, (160,160), interpolation=cv2.INTER_AREA).astype(np.float32)/255
    # RGB and HSV distinguish sky, grass, brass, fur and painted machines.
    r,g,b = a.transpose(2,0,1)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    h,s,v = hsv.transpose(2,0,1)
    # Equal-channel brightness avoids making green foliage brighter by convention.
    gray = a.mean(axis=2)
    # These hue intervals represent sky cyan/blue, grass green, warm brass and brown fur.
    masks = dict(blue=(h>175)&(h<260)&(s>.18)&(v>.35), green=(h>65)&(h<170)&(s>.20),
                 gold=(h>28)&(h<65)&(s>.35)&(v>.35), brown=(h>8)&(h<48)&(s>.22)&(v<.65),
                 red=((h<18)|(h>340))&(s>.50)&(v>.35), orange=(h>10)&(h<40)&(s>.55)&(v>.5),
                 white=(s<.20)&(v>.72), black=v<.22, neutral=s<.17,
                 skin=(r>g*1.08)&(g>b*1.10)&(r>.42)&(s<.6))
    f={}
    for k,m in masks.items():
        # Global, central and upper-field proportions separate object from landscape context.
        f[k]=float(m.mean())
        f[k+'c']=float(m[32:128,32:128].mean()) # Central three-fifths isolates the likely object from borders.
        f[k+'top']=float(m[:60].mean()) # Top three-eighths captures sky or background.
        f[k+'bottom']=float(m[100:].mean()) # Bottom three-eighths captures ground and machine bases.
    # Sobel derivatives separate rectilinear machines/buildings from curved organic scenes.
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/4
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/4 # Three-tap Sobel vertical derivative, divided by four for unit contrast scale.
    mag=np.hypot(gx,gy)
    # A contrast difference of about a tenth is an object edge rather than JPEG noise.
    edge=mag>.10
    f['edge']=float(edge.mean())
    # Near-axis gradients indicate horizontal/vertical contours within roughly 15 degrees.
    axis=(np.minimum(abs(gx),abs(gy))<.27*np.maximum(abs(gx),abs(gy)))&edge
    f['axis']=float(axis.sum()/max(edge.sum(),1))
    f['vertical']=float(((abs(gx)>2*abs(gy))&edge).mean())
    f['horizontal']=float(((abs(gy)>2*abs(gx))&edge).mean())
    # Local residual isolates fur, scales, grass and golf dimples from smooth sky and panels.
    residual=gray-cv2.GaussianBlur(gray,(0,0),1)
    f['fine']=float(np.mean(abs(residual)))
    f['smooth']=float((abs(residual)<.015).mean()) # Residual under 1.5 percent brightness is smooth sky or a panel.
    f['contrast']=float(gray.std())
    f['saturation']=float(s.mean())
    # Opposing dark/white patches are common on springer coats, and brown is their fur colour.
    nearwhite=cv2.dilate(masks['white'].astype(np.uint8),np.ones((7,7),np.uint8))>0
    f['brownwhite']=float((nearwhite&masks['brown']).mean())
    f['blackwhite']=float((nearwhite&masks['black']).mean())
    # White object against green lawn is a characteristic golf arrangement.
    f['golfcontext']=f['whitec']*f['green']
    # Sky in upper field with stronger colour low down separates landscapes from airborne canopy scenes.
    f['sky']=f['bluetop']
    return f


def scores(f):
    # Evidence weights below are hand-chosen ordinal votes: strong object clues get several votes,
    # broad context gets one or two; proportions are scaled to make a quarter-image patch meaningful.
    s={k:0.0 for k in CLASSES}
    s['tench']=2*f['skin']+3*f['brownc']+f['green']+2*(1-f['saturation'])-3*f['axis'] # Fish: skin and central olive/brown mass, outdoor context, nonrectilinear shape; 1-3 votes.
    s['english_springer']=8*f['brownwhite']+5*f['blackwhite']+2*f['whitec']+2*f['brownc']+f['green'] # Dog: white near brown/black fur outweighs broader coat colours, with small grass vote.
    s['cassette_player']=3*f['neutral']+2*f['blackc']+3*f['axis']-2*f['green']-2*f['sky'] # Player: neutral colour and straight contours dominate; outdoor context subtracts votes.
    s['chain_saw']=5*f['orange']+3*f['red']+f['green']+f['blackc']+f['edge'] # Saw: vivid orange/red housing dominates; vegetation and dark blade contribute weak votes.
    s['church']=3*f['axis']+2*f['sky']+2*f['vertical']+f['brown']-f['blackc'] # Church: upright straight walls plus sky and masonry, with dark foreground penalty.
    s['french_horn']=7*f['goldc']+3*f['gold']+f['black']-f['green'] # Horn: central warm brass gets seven votes, distributed brass three; grass is counterevidence.
    s['garbage_truck']=2*f['axis']+2*f['edge']+f['green']+f['white']+f['red'] # Truck: straight contours, busy details, coloured livery and outdoor context each vote.
    s['gas_pump']=3*f['axis']+2*f['whitec']+2*f['red']+f['blackwhite']-f['green'] # Pump: straight white cabinet plus red livery and dark display, with vegetation penalty.
    s['golf_ball']=6*f['golfcontext']+2*f['greenc']+2*f['whitec']+2*f['smooth']-f['edge'] # Golf: white-on-grass interaction gets six votes; smooth round surface and lawn two each.
    s['parachute']=6*f['blue']+2*f['sky']+f['smooth']-f['axis'] # Parachute: broad blue sky dominates, smooth background adds one, straight structure subtracts.
    return s


def predict(path: str) -> str:
    # PIL is the only image reader; RGB conversion makes grayscale and CMYK JPEGs consistent.
    with Image.open(path) as im:
        rgb=np.asarray(im.convert('RGB'))
    # Highest semantic evidence wins; no stored examples or learned parameters are consulted.
    return max(scores(features(rgb)).items(),key=lambda kv:kv[1])[0]

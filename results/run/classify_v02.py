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
    # A little gap repair followed by a tenth-frame opening identifies long straight object contours.
    he=((abs(gy)>2*abs(gx))&edge).astype(np.uint8)
    # Orthogonal directional mask captures upright cabinets and building walls.
    ve=((abs(gx)>2*abs(gy))&edge).astype(np.uint8)
    # Three-pixel closing allows interrupted borders; sixteen pixels is one tenth-frame.
    hl=cv2.morphologyEx(cv2.morphologyEx(he,cv2.MORPH_CLOSE,np.ones((1,3),np.uint8)),cv2.MORPH_OPEN,np.ones((1,16),np.uint8))
    # The same treatment of vertical boundaries makes orientation comparisons fair.
    vl=cv2.morphologyEx(cv2.morphologyEx(ve,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8)),cv2.MORPH_OPEN,np.ones((16,1),np.uint8))
    # Line fractions quantify rigidity; layout compares lower chassis/facade with upper sky/cab.
    f['hline']=float(hl.mean())
    f['vline']=float(vl.mean())
    f['lines']=float((hl.sum()+vl.sum())/max(edge.sum(),1))
    # Half-frame comparisons capture a ground/sky transition rather than an isolated edge.
    f['facade']=float(edge[80:].mean()-edge[:80].mean())
    # Chassis is horizontal low down, with vertical cab contours above it.
    f['chassis']=float(hl[80:].mean()-hl[:80].mean()+vl[:80].mean()-vl[80:].mean())
    # Olive is a muted red-green material unlike saturated foliage and warm red dog fur.
    olive=(np.minimum(r,g)>b+.035)&(abs(r-g)<.12)&(v>.13)&(v<.72)
    f['olive']=float(olive[32:128,16:144].mean()) # Broad middle-body window retains horizontal fish shapes.
    # Material-specific residual separates smooth scales from fine foliage, and brass tubes from warm flat walls.
    f['olivefine']=float((abs(residual)*olive).sum()/max(olive.sum(),1))
    f['golddetail']=float((abs(residual)*masks['gold']).sum()/max(masks['gold'].sum(),1))
    # Coherent orange regions favour chainsaw housings rather than speckled autumn backgrounds.
    n,labels,stats,centers=cv2.connectedComponentsWithStats(masks['orange'].astype(np.uint8),8)
    f['orangeblob']=float(stats[1:,cv2.CC_STAT_AREA].max()/160**2) if n>1 else 0.0 # Normalise by image area.
    # Low-field sky distinguishes airborne subjects from churches beneath sky.
    f['sky']=f['bluetop']
    return f


def scores(f):
    # A semantic vote saturates when the named physical cue is clearly visible; no population statistics set its scale.
    def unit(x):
        # Clipping prevents one broad colour patch from overwhelming independent object cues.
        return float(np.clip(x,0,1))
    # A tenth-frame long-edge occupancy and half of edges in straight runs represent strong rigid structure.
    hor=unit(f['hline']/.10)
    vert=unit(f['vline']/.10) # Upright cabinets are measured on the same tenth-frame scale.
    rigid=unit(f['lines']/.50) # Half of visible edges forming long runs means a strongly manufactured object.
    # A fifth-frame lower-detail increase is a substantial sky-to-facade transition.
    facade=unit(f['facade']/.20)
    # These fractional areas denote large visible material patches: a fifth white/dark coat and a quarter olive body.
    coat=unit(f['whitec']/.20)*unit((f['brownc']+f['blackc'])/.20)
    olive=unit(f['olive']/.25) # A quarter of the broad middle window is a substantial fish body.
    # A few percent adjacent coat colours indicate distinct brown/black and white patches, rather than global cast.
    adjacent=unit((f['brownwhite']+f['blackwhite'])/.04)
    # Scale residual under four percent supports smooth olive scales; multiply by material so blank sky earns nothing.
    fishsmooth=olive*(1-unit(f['olivefine']/.04))
    # Brass needs both warm material and the local contrast of narrow reflective tubes.
    brass=unit(f['goldc']/.20)
    brassdetail=brass*unit(f['golddetail']/.04) # Four percent residual is clearly resolved tubular/specular contrast.
    s={k:0.0 for k in CLASSES}
    # Scores below count roughly five independent votes; two-vote clues are object materials, context gets one.
    s['tench']=2*olive+fishsmooth+unit(f['skin']/.15)+unit(f['green']/.30)-rigid
    # Coat adjacency is stronger than grass; rigid structure subtracts a full vote to reject white machines.
    s['english_springer']=2*coat+2*adjacent+unit(f['brownc']/.25)-2*rigid
    # Audio players combine neutral panels with long horizontal cabinet lines, rarely foliage or a sky transition.
    s['cassette_player']=2*rigid+hor+unit(f['neutral']/.60)+unit(f['blackc']/.30)-facade-unit(f['green']/.15)
    # Orange housing earns three votes only when coherent; broad orange contributes one and dark material another.
    s['chain_saw']=3*unit(f['orangeblob']/.04)+unit(f['orange']/.10)+unit(f['blackc']/.25)-rigid
    # Church evidence combines a detailed lower facade, upright architecture and sky above, with reduced horizontal-panel support.
    s['church']=2*facade+vert+unit(f['sky']/.40)+unit(f['brown']/.25)-hor
    # Fine warm brass dominates a horn decision; grass and rectilinear architecture are contrary evidence.
    s['french_horn']=2*brass+3*brassdetail-unit(f['green']/.15)-rigid
    # Truck chassis has horizontal lower contours, a busy rigid body and modest outdoor context.
    s['garbage_truck']=2*unit(f['chassis']/.10)+hor+rigid+unit(f['edge']/.40)+unit(f['green']/.20)-facade
    # Pump cabinets favour vertical over horizontal long edges and white or red enamel.
    s['gas_pump']=2*rigid+2*vert-hor+unit(f['whitec']/.30)+unit(f['red']/.15)-facade
    # Golf requires white surface and green lawn together, few rigid contours, and comparatively smooth surface detail.
    s['golf_ball']=2*unit(f['green']/.30)*unit(f['whitec']/.15)+unit(f['greenc']/.30)+unit(f['smooth']/.70)+unit(f['whitec']/.25)-2*rigid
    # Aerial canopies receive stronger evidence for low-field sky than sky solely above a building.
    s['parachute']=2*unit(f['blue']/.50)+2*unit(f['bluebottom']/.40)+unit(f['smooth']/.70)-rigid-2*facade
    return s


def predict(path: str) -> str:
    # PIL is the only image reader; RGB conversion makes grayscale and CMYK JPEGs consistent.
    with Image.open(path) as im:
        rgb=np.asarray(im.convert('RGB'))
    # Highest semantic evidence wins; no stored examples or learned parameters are consulted.
    return max(scores(features(rgb)).items(),key=lambda kv:kv[1])[0]

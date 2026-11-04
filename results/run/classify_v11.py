"""Analytic colour, texture and geometry evidence; all constants chosen by visual meaning."""
import numpy as np
from PIL import Image
import cv2

CLASSES = ('tench','english_springer','cassette_player','chain_saw','church','french_horn','garbage_truck','gas_pump','golf_ball','parachute')


def machine_features(rgb):
    # Preserve shape while shrinking the longest side to 192 pixels; this resolves panels without square stretching.
    height,width = rgb.shape[:2]
    # The physical size of every pixel feature is referenced to a common longest image dimension.
    scale = 192/max(height,width)
    # Rounding keeps the shape-preserving thumbnail on integer pixel coordinates.
    a = cv2.resize(rgb,(max(1,round(width*scale)),max(1,round(height*scale))),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    # Equal-channel brightness separates dark audio panels from white dispenser enamel.
    gray = a.mean(axis=2)
    # Channel spread tolerates a fifth-range cast on nominally white paint or neutral equipment.
    spread = a.max(axis=2)-a.min(axis=2)
    r,g,b = a.transpose(2,0,1)
    # A third-range brightness captures deep gray/black audio cabinets and displays rather than merely shaded white paint.
    dark = (gray<1/3)&(spread<.20)
    # Bright neutral enamel includes silver-white pumps while excluding pale blue sky and brass.
    white = (gray>.60)&(spread<.20)
    # Strong red, blue or green enamel provides coloured cabinet candidates; visible channel gaps reject foliage-gray casts.
    paint = ((r>g+.15)&(r>b+.15)&(r>.4))|((b>r+.15)&(b>g+.05)&(b>.4))|((g>r+.10)&(g>b+.10)&(g>.35))
    # Close three-pixel interruptions from labels and reflections without bridging separate large objects.
    closed_dark = cv2.morphologyEx(dark.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    # Morphological component geometry requires no stored prototypes and adapts to the current image.
    f = dict(wide_panel=0.0,tall_enamel=0.0,tall_white=0.0,tall_paint=0.0,tall_mixed=0.0,panel_area=0.0)
    # Components are checked for three semantic materials; the mask count is constant and each is computed just once.
    # Mixed white/red/blue enamel belongs to one cabinet despite painted colour bands; foliage-green is excluded.
    mixed = white|(paint&~((g>r)&(g>b)))
    # The same component geometry tests the full mixed-enamel cabinet as well as its separate materials.
    for name,mask in [('dark',closed_dark),('white',white.astype(np.uint8)),('paint',paint.astype(np.uint8)),('mixed',mixed.astype(np.uint8))]:
        # Labels split physically separate coherent material regions in the current photograph.
        count,labels,stats,centers = cv2.connectedComponentsWithStats(mask,8)
        # Candidate dimensions and areas are vectorised over components rather than over pixels.
        box = stats[1:].astype(np.float32)
        if len(box)==0:
            continue
        # Stats columns are x,y,width,height,area; actual image aspect is retained in the ratio.
        xx,yy,ww,hh,area = box.T
        # Material occupancy within the axis-aligned box rewards solid panels over branching vegetation and coats.
        fill = area/np.maximum(ww*hh,1)
        # Boxes at the frame boundary are often backgrounds, so give them half the closed-object evidence.
        interior = np.where((xx>0)&(yy>0)&(xx+ww<a.shape[1])&(yy+hh<a.shape[0]),1,.5)
        # An eighth of thumbnail area denotes a substantial cabinet panel rather than a logo or tiny display.
        size = np.minimum(area/(a.shape[0]*a.shape[1]/8),1)
        # Components near the centre are more likely the photographed machine than surrounding signs and walls.
        cx,cy = centers[1:].T
        # The broad half-frame radius makes centre preference gentle and independent of a particular composition.
        central = np.maximum(0,1-np.hypot(cx/a.shape[1]-.5,cy/a.shape[0]-.5))
        # A wide audio faceplate starts at a 3:2 ratio and becomes fully elongated at 5:2.
        wide = np.clip(ww/np.maximum(hh,1)-1.5,0,1)
        # Upright dispenser paint starts at 5:4 height-to-width and reaches full support at 9:4.
        tall = np.clip(hh/np.maximum(ww,1)-1.25,0,1)
        # Tiny hairline components receive no panel evidence; a twelfth-frame is a visibly broad material surface.
        broad = np.minimum(np.minimum(ww,hh)/(192/12),1)
        # Rectangularity, size, centrality and broadness must agree rather than compensating for one another.
        evidence = fill*size*interior*central*broad
        if name=='dark':
            # A wide dark region large enough to be the entire object supports a player over a small pump display.
            values = evidence*wide
            f['wide_panel'] = float(values.max())
            # Report the winning panel area to distinguish a resolved object from tiny display fragments.
            idx = int(np.argmax(values))
            f['panel_area'] = float(area[idx]/(a.shape[0]*a.shape[1]))
        else:
            # Tall neutral/coloured panels support dispenser cabinets but need rigid-structure corroboration externally.
            value = float((evidence*tall).max())
            f['tall_'+name] = value
            f['tall_enamel'] = max(f['tall_enamel'],value)
    # Standard finite differences measure repeated straight controls at the same aspect-preserving scale.
    gx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8
    # The perpendicular gradient distinguishes horizontal dial rows from vertical cabinet sides.
    gy = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8
    # A twentieth-range derivative is a visible panel boundary, with a 2:1 orientation criterion allowing modest perspective.
    horizontal = ((abs(gy)>.05)&(abs(gy)>2*abs(gx))).astype(np.uint8)
    # A sixth of the image width is a long control row, grille line or deck edge rather than isolated letters.
    length = max(3,round(a.shape[1]/6))
    # Three-pixel closing repairs little gaps before selecting continuous horizontal traces.
    runs = cv2.morphologyEx(cv2.morphologyEx(horizontal,cv2.MORPH_CLOSE,np.ones((1,3),np.uint8)),cv2.MORPH_OPEN,np.ones((1,length),np.uint8))
    # The middle half-frame emphasises faceplate controls, avoiding roof/sign bands at image boundaries.
    lo,hi = a.shape[0]//4,3*a.shape[0]//4
    # Row support is the fraction of width occupied by long horizontal controls.
    profile = runs[lo:hi].mean(axis=1)
    # Three-pixel row dilation merges the two contrast sides of one thick control into a single band.
    bands = (profile>1/4).astype(np.uint8)
    # A one-dimensional connected-component count measures distinct rows rather than edge pixel totals.
    n,_,st,_ = cv2.connectedComponentsWithStats(bands[:,None],8)
    # Four supported rows mean repeated controls or grille geometry; capping avoids rewarding arbitrary textural excess.
    f['control_rows'] = min((n-1)/4,1)
    # Repeated rows matter only when substantial long horizontal line area is present in the faceplate region.
    f['control_support'] = float(profile.mean())
    return f


def machine_adjustments(f):
    # A quarter-unit of conjunctive component support means several geometric/material requirements agree visibly.
    panel = min(f['wide_panel']/.25,1)
    # The same quarter-unit scale measures a coherent, broad, upright enamel cabinet.
    upright = min(f['tall_enamel']/.25,1)
    # Four distinct supported control rows contribute one vote; a wide panel supplies independent object-shape evidence.
    return dict(cassette_player=panel+f['control_rows'],gas_pump=2*upright)


def geometry_features(rgb):
    # A 128-pixel square supports object-scale line and cabinet analysis at little cost.
    a = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    # Unit RGB gives brightness and colour tolerances a direct fraction-of-range meaning.
    af = a.astype(np.float32) / 255.0
    # Equal-channel brightness avoids colour-specific gradient emphasis for grayscale machinery.
    gray = af.mean(axis=2)
    # The analytic Sobel stencil averages across the edge; eight is its absolute coefficient mass.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8
    # The perpendicular gradient measures horizontal boundaries in the same intensity units.
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8
    # A twentieth-range local difference is a visible edge rather than small JPEG fluctuations.
    edge = np.hypot(gx, gy) > .05
    # Twice the competing derivative allows about a quarter-turn's third of perspective tolerance.
    he = ((np.abs(gy) > 2 * np.abs(gx)) & edge).astype(np.uint8)
    # The transposed criterion isolates upright walls, display sides and cabinet uprights.
    ve = ((np.abs(gx) > 2 * np.abs(gy)) & edge).astype(np.uint8)
    # Closing three-pixel holes repairs little interruptions in the same straight physical boundary.
    hc = cv2.morphologyEx(he, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))
    # The same operation along columns repairs vertical edges.
    vc = cv2.morphologyEx(ve, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))
    # A tenth-frame straight run is substantial machine/building structure, longer than individual fur edges.
    hl = cv2.morphologyEx(hc, cv2.MORPH_OPEN, np.ones((1, 12), np.uint8))
    # Vertical runs use the same physical minimum length.
    vl = cv2.morphologyEx(vc, cv2.MORPH_OPEN, np.ones((12, 1), np.uint8))
    # A tiny denominator guard handles blank photographs without adding meaningful evidence.
    den = float(edge.sum()) + 1e-6
    # Manufactured objects concentrate boundaries along axes and into coherent straight runs.
    f = dict(axis=float((he.sum()+ve.sum())/den), line=float((hl.sum()+vl.sum())/den),
             hline=float(hl.mean()), vline=float(vl.mean()))
    # Relative line orientation separates broad decks/truck bodies from tall dispenser cabinets.
    f['upright'] = float(vl.sum() / (vl.sum()+hl.sum()+1e-6))
    # Row coordinates are dimensionless fractions, so a displacement can mean a fraction of image height.
    y = np.arange(128, dtype=np.float32)[:,None] / 128
    # Horizontal chassis edges tend to lie below upright cabin edges, unlike uniform audio faceplates.
    hy = float((hl*y).sum() / (hl.sum()+1e-6))
    # The vertical-edge centroid locates upright cab, pump and wall structure.
    vy = float((vl*y).sum() / (vl.sum()+1e-6))
    # This signed difference isolates cab-above-chassis layout independent of total boundary strength.
    f['chassis_offset'] = hy-vy
    # Sky-to-facade transition raises lower-half detail in church views; half-frame split is geometric.
    f['lowerdetail'] = float(edge[64:].mean()-edge[:64].mean())
    # Lower straight structure helps reject upper trees as the cause of the detail transition.
    f['lowerlines'] = float(hl[64:].mean()+vl[64:].mean()-hl[:64].mean()-vl[:64].mean())
    # Neutral material permits mild colour cast, while one-third brightness describes a dark audio panel.
    neutral = af.max(axis=2)-af.min(axis=2) < 1/8
    # Dark central material is common to black/silver audio decks but is not independently enough to classify them.
    dark = (gray < 1/3) & neutral
    # The central half-frame suppresses bright photographic borders around a player.
    f['darkcentre'] = float(dark[32:96,32:96].mean())
    # Standard Canny thresholds at one fifth and two fifths byte-range isolate substantial cabinet outlines.
    edges = cv2.Canny(np.uint8(gray*255), 50, 100)
    # Hough segments a fifth-frame long, with small gaps, are possible upright cabinet edges.
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=25, maxLineGap=5)
    f['cabinet'] = 0.0
    f['display'] = 0.0
    if lines is None:
        return f
    # At most eighty returned edges bound the pairwise geometric work even in detailed facades.
    q = lines.reshape(-1,4)[:80].astype(np.float32)
    # Limit slope to one horizontal step per three vertical steps, allowing perspective on an upright pump.
    upright = np.abs(q[:,2]-q[:,0])*3 < np.abs(q[:,3]-q[:,1])
    q = q[upright]
    if len(q) < 2:
        return f
    # Each upright is represented by its mean horizontal position and vertical extent.
    x = (q[:,0]+q[:,2])/2
    # Sorting the endpoint heights makes segment order irrelevant.
    top, bottom = np.minimum(q[:,1],q[:,3]), np.maximum(q[:,1],q[:,3])
    # Pairs of upright sides imply a common rectangle only where their heights overlap.
    x0, x1 = np.minimum(x[:,None],x[None,:]), np.maximum(x[:,None],x[None,:])
    # The overlap excludes mismatched unrelated edges at distant heights.
    y0, y1 = np.maximum(top[:,None],top[None,:]), np.minimum(bottom[:,None],bottom[None,:])
    # Broad upright cabinets are at least an eighth-frame wide and a quarter-frame tall.
    width, height = x1-x0, y1-y0
    # A height comparable to width rejects short horizontal tape-player panels.
    valid = (width>16)&(width<80)&(height>32)&(height>width*.8)
    # Half-frame overlap counts as a full cabinet-side vote; shorter visible cabinets get partial credit.
    extent = np.minimum(np.maximum(height,0)/64,1)
    # Different edge lengths get credit in proportion to their common supported fraction.
    alignment = np.maximum(height,0)/np.maximum(np.minimum((bottom-top)[:,None],(bottom-top)[None,:]),1)
    # Compact upright rectangle support is bounded and does not prefer a specific training image size.
    quality = valid*extent*alignment
    f['cabinet'] = float(quality.max()) if quality.size else 0.0
    # White/coloured panels with an upper dark display are a semantic pump arrangement.
    ii = cv2.integral(dark.astype(np.float32))
    # Rectangle integration measures all candidate displays without per-pixel loops.
    def rectangles(left,right,upper,lower):
        # Clip every bound to the fixed thumbnail extent and keep integer area nonzero.
        l,r = np.clip(left.astype(int),0,127),np.clip(right.astype(int),1,128)
        # The vertical counterpart gives valid integral-image bounds at cropped cabinets.
        t,b = np.clip(upper.astype(int),0,127),np.clip(lower.astype(int),1,128)
        # Four-corner inclusion/exclusion is exact rectangular dark-material occupancy.
        return (ii[b,r]-ii[t,r]-ii[b,l]+ii[t,l])/np.maximum((r-l)*(b-t),1)
    # The central half-width avoids dark cabinet sides and hose hardware.
    left,right = x0+width/4,x1-width/4
    # A display usually occupies the upper-middle third rather than the plinth or the top sign.
    updark = rectangles(left,right,y0+height/6,y0+height/2)
    # The lower half should have less dark neutral area than the digital-display region.
    lowdark = rectangles(left,right,y0+height/2,y1-height/8)
    # A quarter-range occupancy increase is a clear small upper screen over a painted lower body.
    contrast = np.clip((updark-lowdark)/.25,0,1)
    # Cabinet geometry gates display evidence so random dark patches on dogs/foliage cannot count.
    f['display'] = float((quality*contrast).max()) if quality.size else 0.0
    return f


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
    olive=(np.minimum(r,g)>b+.035)&(r>g-.025)&(r<g+.08)&(v>.13)&(v<.72) # Near-equal red/green rejects vivid grass and red-brown fur.
    f['olive']=float(olive[32:128,16:144].mean()) # Broad middle-body window retains horizontal fish shapes.
    # Material-specific residual separates smooth scales from fine foliage, and brass tubes from warm flat walls.
    f['olivefine']=float((abs(residual)*olive).sum()/max(olive.sum(),1))
    f['golddetail']=float((abs(residual)*masks['gold']).sum()/max(masks['gold'].sum(),1))
    # Coherent orange regions favour chainsaw housings rather than speckled autumn backgrounds.
    n,labels,stats,centers=cv2.connectedComponentsWithStats(masks['orange'].astype(np.uint8),8)
    f['orangeblob']=float(stats[1:,cv2.CC_STAT_AREA].max()/160**2) if n>1 else 0.0 # Normalise by image area.
    # Gray cloud and blue sky are smooth bright backgrounds; five-percent derivatives exclude textured panels and leaves.
    blue=(b>r+.04)&(b>g-.03)&(b>.35)
    # Cloud colour is nearly neutral and brighter than midgray; fifteen-percent saturation tolerates haze.
    cloud=(s<.15)&(v>.55)
    # A four-percent derivative describes locally smooth sky rather than resolved object texture.
    skymask=(blue|cloud)&(mag<.04)
    # Lower-half and side-strip sky pool background surrounding an aerial canopy; strips each span one eighth-frame.
    f['skylower']=float(skymask[80:].mean())
    # Pool two side strips of one eighth-frame width to measure background surrounding the object.
    f['skysides']=float(np.concatenate((skymask[:,:20].ravel(),skymask[:,-20:].ravel())).mean())
    # Neutral studio backgrounds share cloudy sky colour, so retain separate lower blue corroboration.
    f['bluelower']=float((blue&(mag<.04))[80:].mean())
    # Bright brass is warm yellow rather than dark red-brown fur; the green/red ratio permits ordinary gold reflections.
    brightgold=(r>b+.12)&(g>b+.06)&(g>.40)&(g>r*.70)&(g<r*1.05)
    f['brightgold']=float(brightgold.mean())
    # Conditional texture separates resolved warm tube highlights from broad warm walls and animal material.
    f['brightfine']=float((abs(residual)*brightgold).sum()/max(brightgold.sum(),1))
    # Smooth lawn reflects shallow depth of field around a golf ball; green has a visible excess over both other primaries.
    lawn=(g>r+.03)&(g>b+.03)&(mag<.05)
    f['smoothgreen']=float(lawn.mean())
    # A broad neutral-bright surface can support the lawn context even when the ball is partly shaded.
    f['ballwhite']=float(((gray>.50)&(np.ptp(a,axis=2)<.20)).mean())
    # White fur interiors over half brightness and under one-sixth chroma admit mild shadow but exclude coloured ears.
    coatwhite=(gray>.55)&(np.ptp(a,axis=2)<.16)
    # Muted warm brown denotes liver coat, with red-green ordering and less than a third-range chroma to reject enamel.
    coatbrown=(r>g*1.10)&(g>b*1.08)&(gray>.08)&(gray<.55)&(np.ptp(a,axis=2)>.04)&(np.ptp(a,axis=2)<.35)
    # Seven-percent-width opening retains actual coat interiors but removes narrow glints, text and brass ridges.
    broadwhite=cv2.morphologyEx(coatwhite.astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    # The same seven-percent contact neighborhood asks for adjoining dark/warm patches rather than unrelated background.
    nearbroad=cv2.dilate(broadwhite,np.ones((11,11),np.uint8))>0
    # Central seven tenths discounts photographic border backgrounds when counting visible coat contact area.
    f['broadadj']=float(((coatbrown|(gray<.20))&nearbroad)[24:136,24:136].sum()/160**2)
    # Pink skin retains blue relative to green unlike brass; tenth-frame area means substantial visible hands.
    pink=(r>g*1.15)&(g>b*1.05)&(b>g*.65)&(r>.45)&(gray>.30)&(np.ptp(a,axis=2)<.40)
    f['pink']=float(pink.mean())
    # A coarser geometry stage supplies screen/cabinet relations and normalized chassis displacement, shared across machine classes.
    geo=geometry_features(rgb)
    f.update(cabinet=geo['cabinet'],display=geo['display'],chassis_offset=geo['chassis_offset'])
    # Aspect-preserving material components and control rows supply corroboration for the player/pump contest.
    f.update(machine_features(rgb))
    # Low-field sky distinguishes airborne subjects from churches beneath sky.
    f['sky']=f['bluetop']
    return f


def structure_scores(f):
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
    adjacent=unit(f['broadadj']/.02) # Two percent image area of contact with broad white coat is clearly visible.
    # Scale residual under four percent supports smooth olive scales; multiply by material so blank sky earns nothing.
    fishsmooth=olive*(1-unit(f['olivefine']/.04))
    # Brass needs both warm material and the local contrast of narrow reflective tubes.
    brass=unit(f['brightgold']/.15) # A sixth-frame of bright warm metal is substantial tube/bell material.
    brassdetail=brass*unit(f['brightfine']/.05) # Five percent residual is clearly resolved tubular/specular contrast.
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
    # Fine bright brass dominates a horn decision; grass, rigid structure and a sky-to-facade transition oppose this reading.
    s['french_horn']=2*brass+3*brassdetail-unit(f['green']/.15)-rigid-2*facade*unit(f['sky']/.30)
    # Truck chassis lies below upright cab contours; a quarter-height centroid displacement earns two votes, gated by rigid structure.
    s['garbage_truck']=2*unit(f['chassis_offset']/.25)*rigid+hor+rigid+unit(f['edge']/.40)+unit(f['green']/.20)-facade
    # Pump cabinets favour vertical long edges, white/red enamel and an upper dark screen over a lighter lower body; display adds one vote.
    s['gas_pump']=2*rigid+2*vert-hor+unit(f['whitec']/.30)+unit(f['red']/.15)+f['display']-facade
    # Golf requires white surface and green lawn together, few rigid contours, and comparatively smooth surface detail.
    s['golf_ball']=2*unit(f['green']/.30)*unit(f['whitec']/.15)+unit(f['greenc']/.30)+unit(f['smooth']/.70)+unit(f['whitec']/.25)-2*rigid
    # A fifth-frame of smooth lawn plus sparse edges and visible neutral white earns two corroborating golf votes.
    s['golf_ball']+=2*unit(f['smoothgreen']/.20)*unit((.30-f['edge'])/.20)*unit(f['ballwhite']/.05)
    # Aerial canopies receive stronger evidence for low-field sky than sky solely above a building.
    s['parachute']=2*unit(f['blue']/.50)+2*unit(f['bluebottom']/.40)+unit(f['smooth']/.70)-rigid-2*facade
    # Surrounding cloudy sky adds two votes, with a small blue region or absence of rigid structure needed for studio-background rejection.
    s['parachute']+=2*unit(f['skylower']/.50)*unit(f['skysides']/.50)*max(unit(f['bluelower']/.10),1-rigid)
    # Once general structure already favours a player or pump, resolve their contest using actual component shape and repeated controls.
    if max(s,key=s.get) in ('cassette_player','gas_pump'):
        # Component/control votes remain internal to this related machine pair, so generic white rectangles cannot override other objects.
        extra=machine_adjustments(f)
        for name,value in extra.items():
            # One panel plus one control vote supports players; two cabinet votes support pumps, as defined above.
            s[name]+=value
    return s


def material_scores(f):
    # Unit-range semantic evidence represents absence through clearly resolved presence, without learned calibration.
    def unit(x):
        # Clipping prevents huge background areas from compensating for a missing essential object property.
        return float(np.clip(x,0,1))
    # A tenth-frame long-boundary occupancy is substantial machine structure; half of all edges forming lines denotes rigidity.
    hor=unit(f['hline']/.10)
    # Upright support uses the same tenth-frame geometric occupancy scale.
    vert=unit(f['vline']/.10)
    # Straight run support is measured relative to half of the visible boundaries.
    rigid=unit(f['lines']/.50)
    # One fifth-frame extra lower detail means a strong sky-to-facade transition.
    facade=unit(f['facade']/.20)
    # A third frame of green means outdoor vegetation; tenth-frame pink means visible hands or a face.
    green=unit(f['green']/.30)
    # Human skin is supporting context for a held fish, not a standalone object cue.
    skin=unit(f['pink']/.10)
    # A quarter of the broad centre occupied by muted olive is a substantial fish-coloured body.
    olive=unit(f['olive']/.25)
    # The scale surface is comparatively smooth at a four-percent brightness-residual scale.
    scales=1-unit(f['olivefine']/.04)
    # Broad adjacent white/dark regions, each fifth-centre area, capture characteristic dog coat patches.
    coat=unit(f['whitec']/.20)*unit((f['brownc']+f['blackc'])/.20)
    # Two percent image area in contact with broad white interiors is resolved coat adjacency.
    adjacent=unit(f['broadadj']/.02)
    # Reflective brass must occupy a sixth-frame region and show five-percent fine highlight variation.
    brass=unit(f['brightgold']/.15)
    # Conditional high-frequency response records tubular/specular material independently of overall area.
    sparkle=unit(f['brightfine']/.05)
    # Most methods below are conjunctions: supporting context cannot replace an absent essential material or geometry cue.
    s={k:0.0 for k in CLASSES}
    # Fish needs olive body material, supported by either human skin or green surroundings and smooth scale material.
    s['tench']=olive*(max(skin,green)+scales)/2*(1-rigid/2)
    # Dog requires pale/dark coat with broad patch contact; dark brown adds support without replacing the coat.
    s['english_springer']=coat*(1+adjacent+unit(f['brownc']/.25))/3*(1-rigid/2)
    # Audio players require rigid broad horizontal structure plus a dark or neutral panel, with little facade/green evidence.
    s['cassette_player']=rigid*max(hor,unit(f['wide_panel']/.25))*max(unit(f['neutral']/.60),unit(f['blackc']/.30))*(1-facade/2)*(1-green/2)
    # Saw needs coherent coloured motor housing, dark equipment and visible contours; housing can be a resolved 1/64-frame patch.
    housing=max(unit(f['orangeblob']/(1/64)),unit(f['orange']/(1/32)))
    # Dark material and fine contours support a compact powered tool rather than a flat orange object.
    s['chain_saw']=housing*(unit(f['blackc']/.25)+unit(f['edge']/.40))/2*(1-rigid/2)
    # Church needs the lower detailed facade and upright structure, supported by sky or masonry colour.
    s['church']=facade*(1+vert)/2*max(unit(f['sky']/.40),unit(f['brown']/.25))
    # Horn requires both yellow metal and fine reflections; foliage, rigidity and a sky/facade arrangement lower confidence.
    s['french_horn']=brass*sparkle*(1-green/2)*(1-rigid/2)*(1-facade*unit(f['sky']/.30)/2)
    # Truck requires rigid horizontal body and cabin-above-chassis layout; an eighth-height separation supplies full layout evidence.
    chassis=unit(f['chassis_offset']/.125)
    # Green/edge details corroborate the outdoors machine, but neither can replace a physical horizontal body.
    s['garbage_truck']=rigid*hor*(1+chassis+unit(f['edge']/.40)+green)/4*(1-facade/2)
    # Pump requires rigid upright cabinet structure plus enamel, with an optional upper screen and tall-component confirmation.
    enamel=max(unit(f['whitec']/.30),unit(f['red']/.15),unit(f['tall_enamel']/.25))
    # A physical upright cabinet must exist; the screen is supporting evidence because some pumps show analog dials.
    s['gas_pump']=rigid*vert*enamel*(1+f['display']+unit(f['tall_enamel']/.25))/3*(1-facade/2)
    # Golf joins white surface with green lawn, supported by smooth blurred vegetation and few rigid outlines.
    white=unit(f['whitec']/.15)
    # Smooth lawn is substantial at one fifth-frame; white can occur anywhere for a small off-centre ball.
    lawn=unit(f['smoothgreen']/.20)*unit(f['ballwhite']/.05)
    # Essential lawn/white composition is distinct from an olive body with human skin.
    s['golf_ball']=max(green*white,lawn)*(1+unit(f['smooth']/.70))/2*(1-rigid/2)
    # Parachute requires smooth sky around the lower and side background; blue corroborates but cloudy sky remains possible.
    surround=unit(f['skylower']/.50)*unit(f['skysides']/.50)
    # Blue low down is strong aerial context, while a neutral surrounding field gets half that context vote.
    s['parachute']=surround*(1+unit(f['bluelower']/.30))/2*(1-rigid/2)*(1-facade)
    return s


def scores(f):
    # First identify classes supported by rigid cabinet/chassis/facade evidence; these parts need not all be visible.
    coarse=structure_scores(f)
    # Architecture and three cabinet/vehicle families share long straight spatial structure.
    winner=max(coarse,key=coarse.get)
    if winner in ('cassette_player','church','garbage_truck','gas_pump'):
        return coarse
    # In less rigid scenes, material conjunctions prevent unrelated context from replacing the object material.
    fine=material_scores(f)
    # Restrict this second stage to the remaining material/aerial/tool families; its unit evidence has a different scale.
    return {k:fine[k] for k in ('tench','english_springer','chain_saw','french_horn','golf_ball','parachute')}


def predict(path: str) -> str:
    # PIL is the only image reader; RGB conversion makes grayscale and CMYK JPEGs consistent.
    with Image.open(path) as im:
        rgb=np.asarray(im.convert('RGB'))
    # Highest semantic evidence wins; no stored examples or learned parameters are consulted.
    return max(scores(features(rgb)).items(),key=lambda kv:kv[1])[0]

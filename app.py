import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.pyplot.switch_backend('Agg') 

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go
import numpy as np

import base64
import io

def image_to_store(image,npix_le):
    out = {'raw':image}
    greyscale = image.convert('LA')
    out['greyscale'] = greyscale
    long_edge = np.argmax(greyscale.size)
    le_length = np.max(greyscale.size)
    se_length = np.min(greyscale.size)

    le_x = np.linspace(0,le_length,npix_le)
    np_y = int(se_length/(le_x[1]-le_x[0]))
    se_x = np.linspace(0,se_length,np_y)
    px_sz = le_x[1]-le_x[0]

    if long_edge == 0:
        gx,gy = np.meshgrid(le_x,se_x)
        npx = npix_le
        npy = np_y
    else:
        gx,gy = np.meshgrid(se_x,le_x)
        npx = np_y
        npy = npix_le
    gx = gx.ravel()
    gy = gy.ravel()

    rs = np.array(greyscale.resize((npx,npy),Image.ANTIALIAS))[:,:,0]
    ps = rs - rs.min()
    ps = 1. - ps/ps.max()

    out['x'] = gx
    out['y'] = gy
    out['ps'] = ps

    return out

app = dash.Dash(__name__)

app.layout = html.Div(
    id="app-container",
    children=[
        html.Div(
            id='banner',
            className='twelvecolumns',
            children=[
                html.H1('PerfScreen Mock-Up App:')
            ]
        ),
        html.Div(children=[
            html.Div(
                id='inputs',
                className='four columns',
                children=[
                    dcc.Upload(
                        id='image-upload',
                        children=html.Div([
                            'Drop or ',
                            html.A('Select Feature Data')
                        ]),
                        style={
                            'width': '99%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '5px'
                        },
                        multiple=False,
                        accept='image/*'
                    ),
                    html.Center(html.H6('Dot Density:')),
                    dcc.Slider(
                        id='dot-slider',
                        min=25,
                        max=150,
                        step=1,
                        value=75
                    ),
                    html.Center(html.H6('Dot Size:')),
                    dcc.Slider(
                        id='dot-size',
                        min=2,
                        max=30,
                        step=1,
                        value=5
                    )
                ]
            ),
            html.Div(
                id='outputs',
                className='eight columns',
                children=[
                    html.Img(
                        id='cur_plot',src=''
                    )
                ]
            )
        ])
    ]
)

@app.callback(
    Output('cur_plot','src'),
    Input('image-upload','contents'),
    Input('dot-slider','value'),
    Input('dot-size','value')
)
def parse_image_upload(image,nple,ds,fgw=10):
    if image is None:
        raise dash.exceptions.PreventUpdate

    ms = int(ds)
    content_type, content_string = image.split(",")
    msg = base64.b64decode(str(content_string))
    img = io.BytesIO(msg)
    img = Image.open(img)

    data = image_to_store(img,int(nple))
    imsz = data['raw'].size
    aspct = imsz[0]/imsz[1]
    fgy = fgw/aspct

    F = plt.figure(figsize=(fgw,fgy))
    ax = F.add_subplot(211)
    ax.imshow(data['raw'],interpolation='None')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = F.add_subplot(212)
    ax.scatter(data['x'],data['y'].max()-data['y'],s=ms*data['ps'],c='k')
    ax.set_aspect('equal')
    ax.set_xlim(0,data['x'].max())
    ax.set_ylim(0,data['y'].max())
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    buf = io.BytesIO()
    F.savefig(buf,format='png')
    plt.close()
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode('ascii').replace("\n", "")

    return "data:image/png;base64,{}".format(out)

if __name__ == '__main__':
    app.run_server(host='127.0.0.1',debug=True)
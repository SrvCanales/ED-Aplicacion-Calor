import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import streamlit.components.v1 as components 
import textwrap
import plotly.graph_objects as go

### AUX

L = 10

# TODO AYUDA PARTE HOMOGENIZACION -------------------------------

if "help_slide" not in st.session_state:
    st.session_state.help_slide = 0

if "help_max_slide" not in st.session_state:
    st.session_state.help_max_slide = 0

# -----------------------------------------------------------------------------
# CSS GENERAL
# -----------------------------------------------------------------------------

st.markdown("""
<style>

/*==========================================================
CONTENEDOR
==========================================================*/

.help-card{
    background:white;
    border-radius:16px;
    border:1px solid #E5E7EB;
    padding:22px;
    margin-top:12px;
    margin-bottom:18px;
    box-shadow:0 4px 12px rgba(0,0,0,.05);
}

/*==========================================================
TÍTULOS
==========================================================*/

.help-title{

    font-size:30px;
    font-weight:700;
    color:#0F172A;

}

.help-subtitle{

    font-size:17px;
    color:#475569;
    margin-bottom:20px;

}

.step-title{

    font-size:26px;
    font-weight:700;
    color:#1E3A8A;
    margin-bottom:10px;

}

.small-title{

    font-size:19px;
    font-weight:600;
    color:#1F2937;

}


/*==========================================================
CAJAS
==========================================================*/

.box-blue{

    background:#EFF6FF;
    border-left:6px solid #2563EB;
    padding:18px;
    border-radius:12px;
    margin-top:12px;
    margin-bottom:18px;

}

.box-green{

    background:#F0FDF4;
    border-left:6px solid #16A34A;
    padding:18px;
    border-radius:12px;
    margin-top:12px;
    margin-bottom:18px;

}

.box-yellow{

    background:#FFF7ED;
    border-left:6px solid #F59E0B;
    padding:18px;
    border-radius:12px;
    margin-top:12px;
    margin-bottom:18px;

}

.box-red{

    background:#FEF2F2;
    border-left:6px solid #DC2626;
    padding:18px;
    border-radius:12px;
    margin-top:12px;
    margin-bottom:18px;

}

.box-gray{

    background:#F8FAFC;
    border:1px solid #E5E7EB;
    padding:18px;
    border-radius:12px;
    margin-top:10px;
    margin-bottom:15px;

}


/*==========================================================
TARJETAS
==========================================================*/

.card{

    background:white;
    border:1px solid #E5E7EB;
    border-radius:15px;
    padding:16px;

}


/*==========================================================
BADGES
==========================================================*/

.badge{

display:inline-block;

padding:5px 12px;

border-radius:999px;

background:#DBEAFE;

color:#1D4ED8;

font-weight:600;

font-size:13px;

margin-bottom:12px;

}

.badge2{

display:inline-block;

padding:5px 12px;

border-radius:999px;

background:#DCFCE7;

color:#15803D;

font-weight:600;

font-size:13px;

margin-bottom:12px;

}


/*==========================================================
NOTAS
==========================================================*/

.note{

font-size:15px;

color:#475569;

line-height:1.75;

}


/*==========================================================
SVG
==========================================================*/

.svg-card{

border:1px solid #E5E7EB;

border-radius:14px;

padding:10px;

background:white;

}


/*==========================================================
BOTONES
==========================================================*/

div.stButton>button{

border-radius:12px;

font-weight:600;

height:44px;

}

</style>
""", unsafe_allow_html=True)


# ==========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS 
# ==========================================
st.set_page_config(page_title="Ruta de Aprendizaje: Calor 1D", layout="wide", initial_sidebar_state="expanded")

# Paleta de Colores Suaves (Material Design)
C_STAT = "#1976D2"  # Azul (Estacionaria)
C_TRANS = "#D32F2F" # Rojo (Transitoria)
C_GEN = "#7B1FA2"   # Morado (Solución General)
C_SPACE = "#388E3C" # Verde (Espacial)
C_TIME = "#F57C00"  # Naranjo (Temporal)

# =============================================================================
# PALETA DE COLORES
# =============================================================================

COLOR_CURVE = "#F57C00"          # Naranja principal
COLOR_CURVE_2 = "#FB8C00"
COLOR_BORDER = "#D84315"         # Rojo cálido
COLOR_BG = "#FFF8F2"             # Fondo crema
COLOR_GRID = "#EADFD3"
COLOR_TEXT = "#3C3C3C"
COLOR_BOX = "#FFF3E0"



# ==========================================
# GESTIÓN DEL ESTADO Y LOGICA DE FLUJO
# ==========================================
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'math_data' not in st.session_state:
    st.session_state.math_data = {}

# Inicializar Inputs en session_state para evitar que se borren al hacer rerun
if 'in_L' not in st.session_state: st.session_state.in_L = "L"
if 'in_alpha' not in st.session_state: st.session_state.in_alpha = "alpha"
if 'in_F' not in st.session_state: st.session_state.in_F = "Q_0 * exp(-t)"
if 'in_A' not in st.session_state: st.session_state.in_A = "T_1"
if 'in_B' not in st.session_state: st.session_state.in_B = "T_2"
if 'in_f' not in st.session_state: st.session_state.in_f = "T_0 * sin(pi*x/L)"

def avanzar():
    st.session_state.step += 1

# ==========================================
# BARRA LATERAL (Progreso)
# ==========================================
st.sidebar.markdown("### 🗺️ Tu Progreso")
pasos = [
    "1. Planteando la EDP",
    "2. Homogeneizando fronteras",
    "3. Valores y funciones propias",
    "4. Expansión por autofunciones",
    "5. Coeficientes temporales",
    "6. Solución General y Simulador"
]

for i, paso in enumerate(pasos):
    if i + 1 < st.session_state.step:
        st.sidebar.markdown(f"✅ <span style='color:gray;'>{paso}</span>", unsafe_allow_html=True)
    elif i + 1 == st.session_state.step:
        st.sidebar.markdown(f"👉 **<span style='color:{C_GEN};'>{paso}</span>**", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"🔒 <span style='color:lightgray;'>{paso}</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Código de Color (Variables):**")
st.sidebar.markdown(f"■ <span style='color:{C_STAT}; font-weight:bold;'>Estacionaria (w)</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"■ <span style='color:{C_TRANS}; font-weight:bold;'>Transitoria (v)</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"■ <span style='color:{C_GEN}; font-weight:bold;'>General (u)</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"■ <span style='color:{C_SPACE}; font-weight:bold;'>Espacial (ϕ)</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"■ <span style='color:{C_TIME}; font-weight:bold;'>Temporal (a)</span>", unsafe_allow_html=True)

# ==========================================
# VARIABLES SIMBÓLICAS BASE
# ==========================================
x, t, tau = sp.symbols('x t tau')
n_sym = sp.Symbol('n', integer=True, positive=True)

# Mapas de nombres personalizados para forzar a SymPy a inyectar el color de LaTeX 
# UNICAMENTE en la letra del símbolo sin pintar operadores ni paréntesis.
COLOR_MAP = {
    'u': rf"\color{{{C_GEN}}}{{u}}",
    'w': rf"\color{{{C_STAT}}}{{w}}",
    'v': rf"\color{{{C_TRANS}}}{{v}}",
    'a': rf"\color{{{C_TIME}}}{{a}}",
    'q': rf"\color{{{C_TIME}}}{{q}}",
    'phi': rf"\color{{{C_SPACE}}}{{\phi}}",
    'lam': rf"\color{{{C_SPACE}}}{{\lambda}}",
    'F_tilde': rf"\color{{{C_TRANS}}}{{\tilde{{F}}}}",
    'f_tilde': rf"\color{{{C_TRANS}}}{{\tilde{{f}}}}"
}

# ==========================================
# MOTOR MATEMÁTICO CENTRAL
# ==========================================
def calcular_matematicas(L_str, alpha_str, F_str, A_str, B_str, f_str):
    """Calcula algebraicamente la solución general utilizando n simbólico."""
    try:
        L = sp.sympify(L_str, locals={'pi': sp.pi})
        alpha = sp.sympify(alpha_str)
        F = sp.sympify(F_str, locals={'x': x, 't': t, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp})
        A = sp.sympify(A_str)
        B = sp.sympify(B_str)
        f = sp.sympify(f_str, locals={'x': x, 'pi': sp.pi, 'sin': sp.sin})
        
        # Homogeneización
        w = sp.simplify(A + (x / L) * (B - A))
        F_tilde = sp.simplify(F - sp.diff(w, t) + alpha**2 * sp.diff(w, x, 2))
        f_tilde = sp.simplify(f - w.subs(t, 0))
        
        # Componente Espacial
        lam_n = n_sym * sp.pi / L
        phi_n = sp.sin(lam_n * x)
        
        # Componente Temporal Proyectada
        q_n_expr = (2/L) * sp.integrate(F_tilde * phi_n, (x, 0, L))
        
        # Simulación numérica (N=6 términos)
        v_sol_num = 0
        for n_val in range(1, 7):
            lam_val = n_val * sp.pi / L
            phi_val = sp.sin(lam_val * x)
            q_n_val = (2/L) * sp.integrate(F_tilde * phi_val, (x, 0, L))
            a_n_0 = (2/L) * sp.integrate(f_tilde * phi_val, (x, 0, L))
            
            factor_k = alpha**2 * lam_val**2
            int_part = sp.integrate(q_n_val.subs(t, tau) * sp.exp(factor_k * tau), tau)
            int_eval = int_part.subs(tau, t) - int_part.subs(tau, 0)
            
            a_n_t = sp.exp(-factor_k * t) * (int_eval + a_n_0)
            v_sol_num += a_n_t * phi_val

        u_final_num = sp.simplify(w + v_sol_num)
        u_num_func = sp.lambdify((x, t), u_final_num, modules=['numpy', 'math'])
        
        st.session_state.math_data = {
            'L': L, 'alpha': alpha, 'F': F, 'A': A, 'B': B, 'f': f,
            'w': w, 'F_tilde': F_tilde, 'f_tilde': f_tilde,
            'lam_n': lam_n, 'phi_n': phi_n, 'q_n_expr': q_n_expr,
            'u_num_func': u_num_func, 'L_num': float(L.evalf())
        }
        return True
    except Exception as e:
        st.error(f"Error en los cálculos matemáticos: {e}")
        return False


# =============================================================================
# FUNKTIONEN
# =============================================================================


# =============================================================================
# ESTILOS
# =============================================================================

def texto(txt):
    st.markdown(
        f"""
<div style="
color:{COLOR_TEXT};
font-size:17px;
line-height:1.8;
text-align:justify;
margin-top:0.3rem;
margin-bottom:0.5rem;
">
{txt}
</div>
""",
        unsafe_allow_html=True
    )


def bloque_latex(titulo, *expresiones):
    """
    Muestra una tarjeta con un título y una o varias expresiones
    matemáticas perfectamente integradas utilizando únicamente
    componentes nativos de Streamlit.
    """

    with st.container(border=True):

        st.markdown(
            f"""
<div style="
color:{COLOR_TEXT};
font-size:17px;
font-weight:600;
margin-bottom:0.3rem;
">
{titulo}
</div>
""",
            unsafe_allow_html=True
        )

        for expr in expresiones:
            st.latex(expr)

def aplicar_estilo(fig):

    fig.update_layout(

        height=340,

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=10
        ),

        paper_bgcolor=COLOR_BG,
        plot_bgcolor=COLOR_BG,

        font=dict(
            family="Arial",
            size=15,
            color=COLOR_TEXT
        ),

        hoverlabel=dict(
            bgcolor=COLOR_BOX,
            font=dict(
                color=COLOR_TEXT,   # <-- Antes aparecía blanco
                size=15
            )
        ),

        showlegend=False
    )

    fig.update_xaxes(

        title="Posición sobre la barra",

        tickmode="array",

        tickvals=[0, L],

        ticktext=["0", "L"],

        range=[-0.3, L+0.3],

        gridcolor=COLOR_GRID,

        linecolor="#7A5230",

        linewidth=2,

        zeroline=False
    )

    fig.update_yaxes(

        title="Temperatura",

        gridcolor=COLOR_GRID,

        linecolor="#7A5230",

        linewidth=2,

        zeroline=True,

        zerolinecolor="#C7B6A3"
    )

    return fig

def grafico_frontera_homogenea():

    x = np.linspace(0, L, 500)

    y = 2.8*np.sin(np.pi*x/L)

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=x,
            y=y,

            mode="lines",

            line=dict(
                color=COLOR_CURVE,
                width=4
            ),

            hovertemplate=(
                "<b>x</b> = %{x:.2f}"
                "<br><b>Temperatura</b> = %{y:.2f} °C"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(

        go.Scatter(

            x=[0, L],

            y=[0, 0],

            mode="markers",

            marker=dict(

                color=COLOR_BORDER,

                size=10,

                line=dict(
                    color="white",
                    width=2
                )
            )
        )
    )

    fig.add_annotation(

        x=0,
        y=0,

        text="<b>0°C</b>",

        showarrow=True,

        yshift=22,

        font=dict(
            color=COLOR_BORDER,
            size=14
        )
    )

    fig.add_annotation(

        x=L,
        y=0,

        text="<b>0°C</b>",

        showarrow=True,

        yshift=22,

        font=dict(
            color=COLOR_BORDER,
            size=14
        )
    )

    fig.add_shape(

        type="line",

        x0=0,
        x1=L,

        y0=-0.35,
        y1=-0.35,

        line=dict(
            color="#B77B40",
            width=9
        )
    )

    fig.add_annotation(

        x=L/2,

        y=-0.7,

        text="<b>Barra</b>",

        showarrow=False,

        font=dict(
            color="#7A5230",
            size=14
        )
    )

    # -------- Restricción de temperatura --------

    fig.update_yaxes(
        range=[0, 3.3]
    )

    return aplicar_estilo(fig)



def grafico_frontera_no_homogenea():

    x = np.linspace(0, L, 500)

    y = 20 + (75-20)*x/L + 6*np.sin(np.pi*x/L)

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=x,
            y=y,

            mode="lines",

            line=dict(

                color=COLOR_CURVE,

                width=4
            ),

            hovertemplate=(
                "<b>x</b> = %{x:.2f}"
                "<br><b>Temperatura</b> = %{y:.2f} °C"
                "<extra></extra>"
            )
        )
    )

    fig.add_trace(

        go.Scatter(

            x=[0, L],

            y=[20, 75],

            mode="markers",

            marker=dict(

                color=COLOR_BORDER,

                size=10,

                line=dict(
                    color="white",
                    width=2
                )
            )
        )
    )

    fig.add_annotation(

        x=0,
        y=20,

        text="<b>20°C</b>",

        showarrow=True,

        yshift=22,

        font=dict(
            color=COLOR_BORDER,
            size=14
        )
    )

    fig.add_annotation(

        x=L,
        y=75,

        text="<b>75°C</b>",

        showarrow=True,

        yshift=22,

        font=dict(
            color=COLOR_BORDER,
            size=14
        )
    )

    fig.add_shape(

        type="line",

        x0=0,
        x1=L,

        y0=0,
        y1=0,

        line=dict(
            color="#B77B40",
            width=9
        )
    )

    fig.add_annotation(

        x=L/2,

        y=-6,

        text="<b>Barra</b>",

        showarrow=False,

        font=dict(
            color="#7A5230",
            size=14
        )
    )

    # -------- Restricción de temperatura --------

    fig.update_yaxes(
        range=[0, 85]
    )

    return aplicar_estilo(fig)


def barra_progreso(): #Progreso homogenización

    progreso=(st.session_state.help_slide+1)/3

    st.progress(progreso)

    c1,c2,c3=st.columns(3)

    with c1:

        if st.session_state.help_slide==0:
            st.success("① ¿Por qué?")
        else:
            st.write("① ¿Por qué? ✓")

    with c2:

        if st.session_state.help_slide==1:
            st.success("② ¿Cómo?")
        elif st.session_state.help_slide>1:
            st.write("② ¿Cómo? ✓")
        else:
            st.write("② ¿Cómo?")

    with c3:

        if st.session_state.help_slide==2:
            st.success("③ ¿Otra sustitución?")
        else:
            st.write("③ ¿Otra sustitución?")


def botones_navegacion(): #Navegación homogenización

    st.divider()

    c1,c2,c3=st.columns([1,4,1])

    with c1:

        if st.session_state.help_slide>0:

            if st.button("⬅ Anterior",use_container_width=True):

                st.session_state.help_slide-=1
                st.rerun()

    with c3:

        if st.session_state.help_slide<2:

            if st.button("Siguiente ➜",use_container_width=True):

                st.session_state.help_slide+=1

                st.session_state.help_max_slide=max(
                    st.session_state.help_slide,
                    st.session_state.help_max_slide
                )

                st.rerun()
                
def svg_frontera_homogenea(): #SVG1, frontera homogenea

    svg="""

<svg width="520" height="250"
xmlns="http://www.w3.org/2000/svg">

<rect width="100%" height="100%" fill="white"/>

<line x1="60" y1="200"
x2="470"
y2="200"
stroke="black"
stroke-width="2"/>

<line x1="60"
y1="200"
x2="60"
y2="30"
stroke="black"
stroke-width="2"/>

<text x="480" y="208"
font-size="20">x</text>

<text x="50" y="25"
font-size="20">u</text>

<text x="45"
y="220"
font-size="18">0</text>

<text x="445"
y="220"
font-size="18">L</text>

<path

d="M60 200
C130 180,
180 90,
260 80
C330 90,
390 180,
460 200"

stroke="#16A34A"

stroke-width="4"

fill="none"/>

<circle cx="60" cy="200" r="5" fill="#16A34A"/>

<circle cx="460" cy="200" r="5" fill="#16A34A"/>

<text

x="150"

y="40"

fill="#15803D"

font-size="20"

font-weight="bold">

Frontera homogénea

</text>

</svg>

"""

    components.html(svg,height=260)

def svg_frontera_no_homogenea(): #SVG 2, fontera no homogenea

    svg="""

<svg width="520" height="250"
xmlns="http://www.w3.org/2000/svg">

<rect width="100%" height="100%" fill="white"/>

<line x1="60" y1="200"
x2="470"
y2="200"
stroke="black"
stroke-width="2"/>

<line x1="60"
y1="200"
x2="60"
y2="30"
stroke="black"
stroke-width="2"/>

<text x="480" y="208"
font-size="20">x</text>

<text x="50"
y="25"
font-size="20">u</text>

<text x="45"
y="220"
font-size="18">0</text>

<text x="445"
y="220"
font-size="18">L</text>

<path

d="M60 150
C140 90,
200 140,
280 120
C360 95,
420 80,
460 70"

stroke="#DC2626"

stroke-width="4"

fill="none"/>

<circle cx="60" cy="150"
r="5"
fill="#DC2626"/>

<circle cx="460"
cy="70"
r="5"
fill="#DC2626"/>

<text

x="105"

y="40"

fill="#B91C1C"

font-size="20"

font-weight="bold">

Frontera no homogénea

</text>

</svg>

"""

    components.html(svg,height=260)

def svg_recta(): #SVG3, recta de homogenización

    svg="""

<svg width="520" height="260"
xmlns="http://www.w3.org/2000/svg">

<rect width="100%" height="100%" fill="white"/>

<line x1="60"
y1="210"
x2="470"
y2="210"
stroke="black"
stroke-width="2"/>

<line x1="60"
y1="210"
x2="60"
y2="30"
stroke="black"
stroke-width="2"/>

<circle
cx="60"
cy="170"
r="5"
fill="#2563EB"/>

<circle
cx="460"
cy="70"
r="5"
fill="#2563EB"/>

<line

x1="60"
y1="170"
x2="460"
y2="70"

stroke="#2563EB"

stroke-width="4"/>

<text
x="95"
y="95"
font-size="18"
fill="#1D4ED8">

w(x)

</text>

<text
x="40"
y="225"
font-size="18">

0

</text>

<text
x="445"
y="225"
font-size="18">

L

</text>

</svg>

"""

    components.html(svg,height=270)

@st.dialog("📖 Profundización matemática: Homogeneización", width="large") ##!!!!
def mostrar_ayuda_profunda(w_d, F_t_d, f_t_d):

    # -------------------------------------------------------------------------
    # TÍTULO
    # -------------------------------------------------------------------------

    st.markdown("""
    <div class="help-title">
    Homogeneización de las condiciones de frontera
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="help-subtitle">

    En esta guía recorrerás paso a paso la idea detrás de la homogeneización.
    La intención no es memorizar una sustitución, sino comprender por qué surge
    naturalmente al resolver la ecuación de calor mediante separación de variables.

    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Barra de progreso
    # -------------------------------------------------------------------------

    barra_progreso()

    st.divider()

    # =========================================================================
    # SLIDE 1
    # =========================================================================

    if st.session_state.help_slide == 0:

        slide_1()

    # =========================================================================
    # SLIDE 2
    # =========================================================================

    elif st.session_state.help_slide == 1:

        slide_2(
            w_d=w_d,
            F_t_d=F_t_d,
            f_t_d=f_t_d
        )

    # =========================================================================
    # SLIDE 3
    # =========================================================================

    else:

        slide_3()

    # -------------------------------------------------------------------------
    # Navegación
    # -------------------------------------------------------------------------

    botones_navegacion()


# =============================================================================
# UTILIDAD
# Tarjetas de información
# =============================================================================

def tarjeta(titulo, texto, color="blue"):

    colores = {
        "blue": "box-blue",
        "green": "box-green",
        "yellow": "box-yellow",
        "red": "box-red",
        "gray": "box-gray"
    }

    st.markdown(
        f"""
<div class="{colores[color]}">

<h4 style="margin-top:0px">
{titulo}
</h4>

<p style="font-size:16px;
line-height:1.8;
margin-bottom:0px">

{texto}

</p>

</div>
""",
        unsafe_allow_html=True
    )


# =============================================================================
# UTILIDAD
# Encabezado de slide
# =============================================================================

def encabezado_slide(numero, titulo, subtitulo):

    st.markdown(
        f"""
<div class="badge">
Paso {numero} de 3
</div>

<div class="step-title">
{titulo}
</div>

<div class="note">

{subtitulo}

</div>
""",
        unsafe_allow_html=True
    )


# =============================================================================
# UTILIDAD
# Cuadro "Idea clave"
# =============================================================================

def idea_clave(texto):

    st.success("💡 Idea clave")

    st.markdown(texto)


# =============================================================================
# UTILIDAD
# Duda desplegable
# =============================================================================

def duda(titulo, contenido):

    with st.expander(
        "❓ " + titulo,
        expanded=False
    ):

        st.markdown(contenido)


# =============================================================================
# UTILIDAD
# Comparación lado a lado
# =============================================================================

def comparacion(col_izq, col_der):

    c1, c2 = st.columns(2)

    with c1:
        col_izq()

    with c2:
        col_der()


# =============================================================================
# UTILIDAD
# Separador visual elegante
# =============================================================================

def separador():

    st.markdown(
        """
<hr style="
margin-top:25px;
margin-bottom:25px;
border:none;
border-top:1px solid #E5E7EB;
">
""",
        unsafe_allow_html=True
    )


# =============================================================================
# MINI TARJETA
# =============================================================================

def mini_card(titulo, cuerpo):

    st.markdown(
        f"""
<div class="card">

<h4 style="margin-bottom:8px">
{titulo}
</h4>

<p style="
font-size:15px;
line-height:1.7;
margin-bottom:0px;
">

{cuerpo}

</p>

</div>
""",
        unsafe_allow_html=True
    )


# =============================================================================
# TARJETA MATEMÁTICA
# =============================================================================

def cuadro_formula(texto_latex):

    st.markdown(
        """
<div class="box-gray">
""",
        unsafe_allow_html=True
    )

    st.latex(texto_latex)

    st.markdown(
        """
</div>
""",
        unsafe_allow_html=True
    )


# =============================================================================
# FUNCIÓN AUXILIAR
# Texto introductorio reutilizable
# =============================================================================

def introduccion(texto):

    st.markdown(
        f"""
<div class="note">

{textwrap.dedent(texto)}

</div>
""",
        unsafe_allow_html=True
    )

def recordatorio(): #HOMOGENIZACION

    st.markdown(
"""
<div class="box-green">

<h4 style="margin-top:0px">

🎯 Idea para recordar

</h4>

<p style="font-size:16px; line-height:1.8;">

La homogeneización no consiste en "inventar" una función.

Consiste en escoger una función conocida que satisfaga las
condiciones de frontera para que la nueva incógnita tenga
fronteras homogéneas.

</p>

</div>
""",
        unsafe_allow_html=True
    )

def mensaje_final(): #HOMOGENIZACION

    st.success("""
¡Excelente!

La homogeneización suele parecer un paso artificial cuando se ve
por primera vez.

Sin embargo, una vez comprendida, se convierte en una herramienta
muy natural: antes de intentar resolver la ecuación diferencial,
eliminamos primero aquello que ya conocemos (las condiciones de
frontera) y concentramos el esfuerzo matemático únicamente en la
parte realmente desconocida del problema.
""")

def slide_1():

    encabezado_slide(

        1,

        "¿Por qué debemos homogeneizar?",

        """
Antes de introducir una sustitución es importante comprender qué
problema estamos intentando resolver.

La homogeneización no es un truco algebraico; aparece porque el
método de separación de variables requiere un tipo muy particular
de condiciones de frontera.
"""
    )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("### 🔥 Fronteras homogéneas")

        st.plotly_chart(
            grafico_frontera_homogenea(),
            use_container_width=True
        )

        texto("""
Una frontera homogénea significa que la temperatura vale
exactamente cero en ambos extremos de la barra.
""")

        bloque_latex(
    "Condiciones de frontera",
    r"u(0,t)=0",
    r"u(L,t)=0"
)

    with c2:

        st.markdown("### 🌡️ Fronteras no homogéneas")

        st.plotly_chart(
            grafico_frontera_no_homogenea(),
            use_container_width=True
        )

        texto("""
En muchos problemas físicos los extremos permanecen a
temperaturas constantes distintas de cero.
""")

        bloque_latex(
    "Condiciones de frontera",
    r"u(0,t)=T_1",
    r"u(L,t)=T_2",
    r"T_1\neq0,\qquad T_2\neq0"
)


    separador()

    tarjeta(

        "¿Por qué esto representa un inconveniente?",

        """
El método de separación de variables construye la solución
como combinación de autofunciones.

Estas funciones deben satisfacer exactamente las mismas
condiciones de frontera que la incógnita.

Cuando las fronteras son homogéneas aparecen familias muy
simples de funciones (senos, cosenos, funciones hiperbólicas,
etc.) que forman una base del espacio solución.

Si las fronteras no son homogéneas, esas funciones dejan de
cumplir las condiciones requeridas y el procedimiento deja de
funcionar directamente.
""",

        "red"
    )

    st.markdown("### ¿Qué ocurre matemáticamente?")

    texto("Supongamos que intentamos escribir la solución como")

    bloque_latex(
    "Separación de variables",
    r"u(x,t)=X(x)\,T(t)"
)


    texto("Si las fronteras son homogéneas obtenemos")

    bloque_latex(
    "Condiciones para la función espacial",
    r"X(0)=0,\qquad X(L)=0"
)


    texto("""
Estas son precisamente las condiciones del problema de
autovalores.

Gracias a ello aparecen las autofunciones que utilizaremos
posteriormente para construir la solución.
""")

    texto("Si en cambio imponemos")

    bloque_latex(
    "Si las fronteras no son homogéneas",
    r"u(0,t)=20,\qquad u(L,t)=75"
)

    texto("la separación")

    bloque_latex(
    "La separación",
    r"u(x,t)=X(x)T(t)"
)
    texto("""
ya no puede satisfacer simultáneamente ambas condiciones
para todo tiempo.

Por ello primero transformamos el problema mediante la
homogeneización.
""")

    separador()

    tarjeta(

        "Interpretación física",

        """
Imagina una barra cuyos extremos permanecen a temperaturas
fijas.

Una parte de la distribución de temperatura simplemente conecta
ambos extremos.

Lo realmente interesante es estudiar cómo evoluciona la
temperatura alrededor de ese estado impuesto.

La homogeneización separa ambas contribuciones.
""",

        "blue"
    )

    idea_clave(
        """
No modificamos la física del problema.

Únicamente cambiamos la función que vamos a resolver para que
sus extremos sean cero.

Gracias a ello podremos expresar posteriormente la solución
como una serie de autofunciones.
"""
    )

    separador()
    
############################
## DUDAS1
############################

    duda(
        "¿Qué significa realmente que una frontera sea homogénea?",
        r"""
Una condición de frontera es homogénea cuando el valor
impuesto sobre la incógnita es exactamente cero.

No importa si la solución toma valores distintos de cero
en el interior del dominio.

Lo único que exige la frontera homogénea es que la función
coincida con cero únicamente en los extremos.
"""
    )

    duda(
        "¿Por qué el cero es tan especial?",
        r"""
No existe nada físicamente especial en el número cero.

Lo importante es que las condiciones homogéneas permiten
construir un problema de autovalores sencillo.

Las funciones

\[
\sin\!\left(\frac{n\pi x}{L}\right)
\]

satisfacen automáticamente

\[
X(0)=X(L)=0.
\]

Eso las convierte en una base natural para representar
la solución.
"""
    )

    duda(
        "¿No sería posible resolver directamente el problema original?",
        r"""
Sí.

Existen métodos para trabajar directamente con condiciones
de frontera no homogéneas.

Sin embargo, el desarrollo suele ser bastante más largo.

La homogeneización transforma el problema en otro equivalente
cuya resolución mediante separación de variables resulta
mucho más sencilla.
"""
    )

    duda(
        "¿La solución cambia después de homogeneizar?",
        r"""
No.

La temperatura física sigue siendo exactamente la misma.

Únicamente cambia la forma de representarla matemáticamente.

Al final del procedimiento recuperaremos la solución original.
"""
    )

def slide_2(w_d, F_t_d, f_t_d):

    # =========================================================================
    # ENCABEZADO
    # =========================================================================

    encabezado_slide(
        2,
        "¿Cómo se homogeneiza un problema?",
        """
Ahora construiremos una nueva incógnita que sí satisfaga fronteras
homogéneas.

La idea consiste en separar la temperatura en una parte conocida,
responsable de las condiciones de frontera, y otra que describa
la evolución restante del sistema.
"""
    )

    # =========================================================================
    # IDEA PRINCIPAL
    # =========================================================================

    tarjeta(
        "🔥 La idea fundamental",
        """
En lugar de resolver directamente la temperatura u(x,t),
la escribiremos como la suma de dos funciones.

Una de ellas será elegida por nosotros para satisfacer
automáticamente las condiciones de frontera.

La otra contendrá toda la información dinámica del problema
y será la nueva incógnita que resolveremos mediante separación
de variables.
""",
        "green"
    )

    separador()

    # =========================================================================
    # SUSTITUCIÓN
    # =========================================================================

    st.markdown("## 🔥 La sustitución")

    bloque_latex(
        "Descomposición de la temperatura",
        rf"""
{COLOR_MAP['u']}(x,t)=
{COLOR_MAP['w']}(x,t)+
{COLOR_MAP['v']}(x,t)
"""
    )

    texto("""
No se trata de una identidad misteriosa.

Simplemente estamos descomponiendo la temperatura en dos partes,
de la misma manera que un vector puede escribirse como suma
de otros dos vectores.

La ventaja es que podremos controlar por separado las
condiciones de frontera y la evolución temporal.
""")

    separador()

    # =========================================================================
    # REPRESENTACIÓN DE w(x,t)
    # =========================================================================

    st.markdown("## 🌡️ ¿Qué representa la función $w(x,t)$?")

    x = np.linspace(0, L, 400)

    T1 = 20
    T2 = 75

    y = T1 + (T2 - T1) * x / L

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(
                color=COLOR_CURVE,
                width=4
            ),
            hovertemplate=
            "<b>x</b> = %{x:.2f}"
            "<br><b>w(x)</b> = %{y:.2f} °C"
            "<extra></extra>"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, L],
            y=[T1, T2],
            mode="markers",
            marker=dict(
                color=COLOR_BORDER,
                size=11,
                line=dict(
                    color="white",
                    width=2
                )
            )
        )
    )

    fig.add_annotation(
        x=0,
        y=T1,
        text="<b>T₁</b>",
        showarrow=True,
        yshift=20,
        font=dict(color=COLOR_BORDER)
    )

    fig.add_annotation(
        x=L,
        y=T2,
        text="<b>T₂</b>",
        showarrow=True,
        yshift=20,
        font=dict(color=COLOR_BORDER)
    )

    fig.add_shape(
        type="line",
        x0=0,
        x1=L,
        y0=0,
        y1=0,
        line=dict(
            color="#B77B40",
            width=9
        )
    )

    fig.add_annotation(
        x=L/2,
        y=-6,
        text="<b>Barra</b>",
        showarrow=False,
        font=dict(
            color="#7A5230",
            size=14
        )
    )

    fig.update_yaxes(range=[0, 85])

    st.plotly_chart(
        aplicar_estilo(fig),
        use_container_width=True
    )

    texto("""
La función <b>w(x,t)</b> conecta exactamente las temperaturas
impuestas en ambos extremos.

Su única misión consiste en satisfacer automáticamente
las condiciones de frontera.

Todavía no hemos resuelto la ecuación diferencial.

Simplemente hemos construido una función conveniente.
""")

    bloque_latex(
        "Una elección habitual para w(x,t)",
        rf"""
{COLOR_MAP['w']}(x,t)=
{sp.latex(w_d)}
"""
    )

    st.info("""
💡 Observa que la función w queda completamente determinada
por las temperaturas impuestas en los extremos.
""")

    separador()

    # =========================================================================
    # COMPARACIÓN
    # =========================================================================

    col1, col2 = st.columns(2)

    with col1:

        mini_card(
            "🔥 Parte estacionaria",
            """
Corresponde a la función w(x,t).

Es completamente conocida.

Su misión consiste únicamente en satisfacer
las condiciones de frontera.

En muchos problemas básicos es simplemente
una recta.
"""
        )

    with col2:

        mini_card(
            "🌡️ Parte transitoria",
            """
Corresponde a la función v(x,t).

Describe la evolución de la temperatura
una vez eliminada la influencia directa
de las fronteras.

Esta será la función que resolveremos
mediante separación de variables.
"""
        )

    separador()

    # =========================================================================
    # CONSECUENCIA
    # =========================================================================

    st.markdown("## 🔥 ¿Qué ocurre con la nueva incógnita?")

    bloque_latex(
        "Relación entre ambas funciones",
        r"""
u=w+v
"""
    )

    bloque_latex(
        "Despejando la nueva incógnita",
        r"""
v=u-w
"""
    )

    texto("""
Si la función w fue construida correctamente,
ocurre algo muy importante.

La nueva incógnita posee automáticamente
fronteras homogéneas.
""")

    bloque_latex(
        "Condiciones de frontera para v",
        r"""
v(0,t)=0
""",
        r"""
v(L,t)=0
"""
    )

    idea_clave(
        r"""
No estamos obligando a la temperatura física
a ser cero.

La que vale cero en los extremos es únicamente
la nueva incógnita

\[
v(x,t).
\]

La temperatura original sigue siendo

\[
u(x,t)=w(x,t)+v(x,t).
\]
"""
    )

    separador()

    # =========================================================================
    # NUEVA ECUACIÓN
    # =========================================================================

    st.markdown("## 🌡️ ¿Qué cambia después de sustituir?")

    texto("""
Al reemplazar

u=w+v

en la ecuación diferencial aparecen tres
modificaciones naturales.
""")

    c1, c2, c3 = st.columns(3)

    with c1:

        bloque_latex(
            "① Fronteras",
            r"""
v(0,t)=0
""",
            r"""
v(L,t)=0
"""
        )

    with c2:

        bloque_latex(
            "② Nueva fuente",
            rf"""
{COLOR_MAP['F_tilde']}(x,t)=
{sp.latex(F_t_d)}
"""
        )

    with c3:

        bloque_latex(
            "③ Nueva condición inicial",
            rf"""
{COLOR_MAP['v']}(x,0)=
{sp.latex(f_t_d)}
"""
        )

    st.success("""
🔥 Todo el efecto de las temperaturas impuestas
en los extremos ha quedado absorbido por la función
w(x,t).

La nueva ecuación ya posee fronteras homogéneas y
está lista para aplicar separación de variables.
""")

    separador()

    # =========================================================================
    # DUDAS2
    # =========================================================================

    duda(
        "¿Por qué se escoge precisamente una recta?",
        r"""
Porque normalmente las temperaturas de los extremos son
constantes.

La recta es la función más sencilla que conecta ambos valores.

Además,

\[
w''(x)=0,
\]

por lo que introduce la menor cantidad posible de términos nuevos
al sustituirla en la ecuación diferencial.

Es, por tanto, la elección más simple y conveniente.
"""
    )

    # -------------------------------------------------------------------------

    duda(
        "¿Qué significa 'solución estacionaria'?",
        r"""
Se denomina estacionaria a una solución que ya no cambia con el
tiempo.

Si esperamos un tiempo suficientemente largo, la temperatura suele
aproximarse a un perfil fijo.

La función \(w\) representa precisamente esa parte permanente del
problema.

La función \(v\) mide únicamente cuánto falta para alcanzar dicho
estado.
"""
    )

    # -------------------------------------------------------------------------

    duda(
        "¿Qué significa 'solución transitoria'?",
        r"""
La solución transitoria describe la evolución temporal.

Al inicio puede tener valores importantes.

Conforme transcurre el tiempo suele hacerse cada vez más pequeña,
hasta desaparecer.

Cuando eso ocurre únicamente permanece la parte estacionaria.
"""
    )

    # -------------------------------------------------------------------------

    duda(
        "¿La sustitución siempre simplifica la ecuación?",
        r"""
Sí en el sentido más importante:

las condiciones de frontera pasan a ser homogéneas.

Aunque puedan aparecer nuevos términos en la ecuación o en la
condición inicial, el problema resultante puede resolverse mediante
separación de variables, lo que compensa ampliamente esas pequeñas
modificaciones.
"""
    )

# =============================================================================
# SLIDE 3 - LABORATORIO DE SUSTITUCIONES
# (Reemplaza el comienzo de slide_3() hasta antes del bloque
# elif tipo=="Dependiente del tiempo")
# =============================================================================

def slide_3():

    encabezado_slide(
        3,
        "Laboratorio de sustituciones",
        """
Hasta ahora hemos utilizado una sustitución lineal.

Sin embargo, no es la única posibilidad.

En esta sección exploraremos brevemente otras alternativas para
comprender que existen muchas funciones capaces de homogeneizar
las condiciones de frontera, aunque para este curso trabajaremos
casi siempre con la opción más sencilla.
"""
    )

    # =========================================================================
    # INTRODUCCIÓN
    # =========================================================================

    tarjeta(
        "🔥 Una misma idea, distintas funciones",
        """
El objetivo de una sustitución siempre es el mismo:

transformar el problema original en otro equivalente con
condiciones de frontera homogéneas.

Existen muchas funciones capaces de lograrlo.

En este curso nos centraremos principalmente en la sustitución
lineal, ya que suele ser la alternativa más simple y eficiente.
""",
        "green"
    )

    separador()

    try:

        tipo = st.segmented_control(
            "Selecciona una sustitución",
            [
                "Lineal",
                "Dependiente del tiempo",
                "Cuadrática",
                "Trigonométrica"
            ],
            default="Lineal"
        )

    except:

        tipo = st.radio(
            "Selecciona una sustitución",
            [
                "Lineal",
                "Dependiente del tiempo",
                "Cuadrática",
                "Trigonométrica"
            ],
            horizontal=True
        )

    separador()

    # =========================================================================
    # LINEAL
    # =========================================================================

    if tipo == "Lineal":

        st.markdown("## 🔥 Sustitución lineal")

        texto("""
La sustitución lineal es la que utilizaremos durante todo el curso.

Conecta las temperaturas impuestas en ambos extremos mediante
la función más sencilla posible.
""")

        col1, col2 = st.columns([1.3,1])

        with col1:

            T1 = 20
            T2 = 75

            x = np.linspace(0, L, 400)

            y = T1 + (T2-T1)*x/L

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(
                        color=COLOR_CURVE,
                        width=4
                    ),
                    hovertemplate=
                    "<b>x</b> = %{x:.2f}"
                    "<br><b>w(x)</b> = %{y:.2f} °C"
                    "<extra></extra>"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[0,L],
                    y=[T1,T2],
                    mode="markers",
                    marker=dict(
                        color=COLOR_BORDER,
                        size=11,
                        line=dict(
                            color="white",
                            width=2
                        )
                    )
                )
            )

            fig.add_annotation(
                x=0,
                y=T1,
                text="<b>T₁</b>",
                showarrow=True,
                yshift=18,
                font=dict(color=COLOR_BORDER)
            )

            fig.add_annotation(
                x=L,
                y=T2,
                text="<b>T₂</b>",
                showarrow=True,
                yshift=18,
                font=dict(color=COLOR_BORDER)
            )

            fig.add_shape(
                type="line",
                x0=0,
                x1=L,
                y0=0,
                y1=0,
                line=dict(
                    color="#B77B40",
                    width=9
                )
            )

            fig.add_annotation(
                x=L/2,
                y=-6,
                text="<b>Barra</b>",
                showarrow=False,
                font=dict(
                    color="#7A5230",
                    size=14
                )
            )

            fig.update_yaxes(range=[0,85])

            st.plotly_chart(
                aplicar_estilo(fig),
                use_container_width=True
            )

        with col2:

            bloque_latex(
                "Sustitución utilizada",
                r"""
w(x)=
T_1+
\frac{T_2-T_1}{L}x
"""
            )

            texto("""
La recta conecta exactamente ambas temperaturas.

No intenta resolver la ecuación diferencial.

Su única misión consiste en satisfacer
las condiciones de frontera.
""")

        separador()

        tarjeta(
            "🌡️ Interpretación",
            """
La función w representa el perfil estacionario más sencillo
compatible con las temperaturas impuestas.

Después de restarla a la temperatura original,
la nueva incógnita será cero en ambos extremos.

Gracias a ello podremos aplicar separación de variables.
""",
            "blue"
        )

        separador()

        st.markdown("### 📋 Resumen")

        c1,c2,c3,c4 = st.columns(4)

        with c1:
            st.metric(
                "Complejidad",
                "⭐"
            )

        with c2:
            st.metric(
                "Uso",
                "Curso"
            )

        with c3:
            st.metric(
                "Álgebra",
                "Muy simple"
            )

        with c4:
            st.metric(
                "Recomendada",
                "✅ Sí"
            )

        st.progress(.18)

        texto("""
Es la elección más sencilla cuando las temperaturas
de los extremos permanecen constantes.
""")

        separador()

        idea_clave(
r"""
Cuando

\[
u(0,t)=T_1,
\qquad
u(L,t)=T_2,
\]

permanecen constantes, una recta ya satisface exactamente
las condiciones de frontera.

No necesitamos una función más complicada.
"""
        )

        duda(
            "¿Por qué elegir precisamente una recta?",
            r"""
Porque buscamos la función más sencilla capaz de cumplir
el objetivo.

Además,

\[
w''(x)=0,
\]

por lo que la ecuación diferencial apenas se modifica.

En Matemáticas suele ser preferible la solución más simple
que resuelva completamente el problema.
"""
        )

        # =============================================================================
# CASOS: Dependiente del tiempo + Cuadrática + Trigonométrica
# (Reemplaza los bloques elif originales)
# =============================================================================

    # =========================================================================
    # DEPENDIENTE DEL TIEMPO
    # =========================================================================

    elif tipo == "Dependiente del tiempo":

        st.markdown("## ⏳ Sustitución dependiente del tiempo")

        tarjeta(
            "Más allá del caso básico",
            """
La idea de la homogeneización no cambia.

La diferencia es que ahora las temperaturas impuestas en los
extremos también evolucionan con el tiempo.

La función w debe adaptarse continuamente para seguir
satisfaciendo las condiciones de frontera.
""",
            "blue"
        )

        col1, col2 = st.columns([1.3,1])

        with col1:

            x = np.linspace(0, L, 400)

            T1 = 10
            T2 = 40

            y = T1 + (T2-T1)*x/L

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(
                        color=COLOR_CURVE,
                        width=4,
                        dash="dash"
                    ),
                    hovertemplate=
                    "<b>x</b> = %{x:.2f}"
                    "<br><b>w(x,t)</b> = %{y:.2f}"
                    "<extra></extra>"
                )
            )

            fig.add_shape(
                type="line",
                x0=0,
                x1=L,
                y0=0,
                y1=0,
                line=dict(
                    color="#B77B40",
                    width=9
                )
            )

            fig.update_yaxes(range=[0,50])

            st.plotly_chart(
                aplicar_estilo(fig),
                use_container_width=True
            )

        with col2:

            bloque_latex(
                "Ejemplo",
                r"""
w(x,t)
=
10e^{-t}
+
\frac{30e^{-t}}{L}x
"""
            )

            texto("""
La función sigue siendo lineal respecto a x,
pero ahora cambia con el tiempo.

En cada instante conecta correctamente
las temperaturas impuestas.
""")

        idea_clave(
r"""
La estrategia es exactamente la misma.

Simplemente permitimos que la función utilizada para
homogeneizar también dependa del tiempo.
"""
        )

        st.info("""
💡 Este tipo de sustituciones aparece cuando las temperaturas
de frontera ya no permanecen constantes.
""")

        separador()

    # =========================================================================
    # CUADRÁTICA
    # =========================================================================

    elif tipo == "Cuadrática":

        st.markdown("## 📈 Sustitución cuadrática")

        col1, col2 = st.columns([1.3,1])

        with col1:

            x = np.linspace(0, L, 400)

            y = 20 + 55*(x/L) + 12*(x/L)*(1-x/L)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(
                        color=COLOR_CURVE,
                        width=4
                    ),
                    hovertemplate=
                    "<b>x</b> = %{x:.2f}"
                    "<br><b>w(x)</b> = %{y:.2f}"
                    "<extra></extra>"
                )
            )

            fig.add_shape(
                type="line",
                x0=0,
                x1=L,
                y0=0,
                y1=0,
                line=dict(
                    color="#B77B40",
                    width=9
                )
            )

            fig.update_yaxes(range=[0,90])

            st.plotly_chart(
                aplicar_estilo(fig),
                use_container_width=True
            )

        with col2:

            bloque_latex(
                "Ejemplo",
                r"""
w(x)
=
Ax^2+Bx+C
"""
            )

            texto("""
La función ya no es una recta.

Ahora posee curvatura, lo que modifica
sus derivadas espaciales.
""")

        tarjeta(
            "¿Por qué alguien usaría una parábola?",
            """
En algunos problemas especiales puede resultar útil para
simplificar ciertos términos de la ecuación diferencial.

Sin embargo, para los problemas introductorios de este curso
normalmente no ofrece ventajas importantes frente a una recta.
""",
            "green"
        )

        idea_clave(
r"""
Una parábola también puede homogeneizar el problema.

Sin embargo, si una recta ya funciona correctamente,
normalmente preferimos la opción más simple.
"""
        )

        st.warning("""
⚠️ En cursos básicos rara vez necesitaremos este tipo de
sustituciones.
""")

        separador()

    # =========================================================================
    # TRIGONOMÉTRICA
    # =========================================================================

    elif tipo == "Trigonométrica":

        st.markdown("## 🌊 Sustitución trigonométrica")

        col1, col2 = st.columns([1.3,1])

        with col1:

            x = np.linspace(0, L, 500)

            y = 20 + 55*x/L + 10*np.sin(np.pi*x/L)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(
                        color=COLOR_CURVE,
                        width=4
                    ),
                    hovertemplate=
                    "<b>x</b> = %{x:.2f}"
                    "<br><b>w(x)</b> = %{y:.2f}"
                    "<extra></extra>"
                )
            )

            fig.add_shape(
                type="line",
                x0=0,
                x1=L,
                y0=0,
                y1=0,
                line=dict(
                    color="#B77B40",
                    width=9
                )
            )

            fig.update_yaxes(range=[0,90])

            st.plotly_chart(
                aplicar_estilo(fig),
                use_container_width=True
            )

        with col2:

            bloque_latex(
                "Ejemplo",
                r"""
w(x)
=
A\sin\!\left(
\frac{\pi x}{L}
\right)+B
"""
            )

            texto("""
Este tipo de funciones aparece cuando la física
del problema presenta comportamientos periódicos
u oscilatorios.
""")

        tarjeta(
            "Una posibilidad más",
            """
Las funciones trigonométricas también pueden utilizarse
para construir sustituciones válidas.

Sin embargo, suelen generar desarrollos algebraicos más
largos y rara vez son necesarias en una primera introducción
a la ecuación de calor.
""",
            "yellow"
        )

        idea_clave(
r"""
La mejor sustitución no es la más sofisticada.

Es la que simplifica el problema de la forma más eficiente.
"""
        )

        st.warning("""
⚠️ En este curso las veremos principalmente como una
curiosidad matemática y no como la estrategia principal.
""")

        separador()



##################################
# CONTENIDOS
##################################

st.title("🔥 Simulador de EDP: Calor 1D Analítico")
st.markdown("Sigue esta ruta guiada interactiva para resolver tu Problema de Valor Inicial y de Frontera (PVIF) paso a paso.")
st.divider()

st.header("1. Planteando la EDP")
st.markdown("Define los parámetros físicos y geométricos de tu sistema unidimensional.")

# Contenedor nativo con enlace directo al session_state para retener los datos ingresados
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.in_L = st.text_input("Longitud de la barra (L):", value=st.session_state.in_L)
        st.session_state.in_alpha = st.text_input("Difusividad térmica (α):", value=st.session_state.in_alpha)
        st.session_state.in_F = st.text_input("Fuente externa de calor F(x,t):", value=st.session_state.in_F)
    with col2:
        st.session_state.in_A = st.text_input("Frontera Izquierda u(0,t):", value=st.session_state.in_A)
        st.session_state.in_B = st.text_input("Frontera Derecha u(L,t):", value=st.session_state.in_B)
        st.session_state.in_f = st.text_input("Perfil Inicial u(x,0):", value=st.session_state.in_f)

transformaciones = (standard_transformations + (implicit_multiplication_application,))

def parsear_seguro(expr_str):
    if not expr_str.strip():
        raise ValueError("Expresión vacía")
    expr = parse_expr(expr_str, transformations=transformaciones)
    reemplazos = {sim: (x if sim.name == 'x' else (t if sim.name == 't' else sim)) for sim in expr.free_symbols}
    return expr.subs(reemplazos)

st.subheader("Modelo Matemático (Tiempo Real)")

try:
    L_s = parsear_seguro(st.session_state.in_L)
    alpha_s = parsear_seguro(st.session_state.in_alpha)
    F_s = parsear_seguro(st.session_state.in_F)
    A_s = parsear_seguro(st.session_state.in_A)
    B_s = parsear_seguro(st.session_state.in_B)
    f_s = parsear_seguro(st.session_state.in_f)
    
    w_dyn = sp.simplify(A_s + (x / L_s) * (B_s - A_s))
    F_t_dyn = sp.simplify(F_s - sp.diff(w_dyn, t) + alpha_s**2 * sp.diff(w_dyn, x, 2))
    f_t_dyn = sp.simplify(f_s - w_dyn.subs(t, 0))
    
    alpha_term = sp.latex(alpha_s**2)
    
    # Construcción precisa de LaTeX inyectando solo el color en el carácter 'u'
    latex_sistema = rf"""
    \begin{{cases}}
    \frac{{\partial {COLOR_MAP['u']}}}{{\partial t}} = {alpha_term} \frac{{\partial^2 {COLOR_MAP['u']}}}{{\partial x^2}} + {sp.latex(F_s)}, & 0 < x < {sp.latex(L_s)}, \quad t > 0 \\[8pt]
    {COLOR_MAP['u']}(0,t) = {sp.latex(A_s)}, & t > 0 \\[8pt]
    {COLOR_MAP['u']}({sp.latex(L_s)},t) = {sp.latex(B_s)}, & t > 0 \\[8pt]
    {COLOR_MAP['u']}(x,0) = {sp.latex(f_s)}, & 0 \le x \le {sp.latex(L_s)}
    \end{{cases}}
    """
    
    with st.container(border=True):
        st.latex(latex_sistema)
        
    col_help, _ = st.columns([1, 2])
    with col_help:
        if st.button("ℹ️ Ver Fundamentos de Homogeneización", use_container_width=True):
            mostrar_ayuda_profunda(w_dyn, F_t_dyn, f_t_dyn)

except Exception:
    with st.container(border=True):
        st.latex(r"\text{Esperando expresiones matemáticas válidas...}")

st.info("💡 **Reto de Aprendizaje:** Identifica mentalmente cuáles términos causan que el problema no sea homogéneo antes de avanzar.")

if st.button("Guardar Parámetros y Avanzar 🚀", type="primary"):
    exito = calcular_matematicas(st.session_state.in_L, st.session_state.in_alpha, st.session_state.in_F, st.session_state.in_A, st.session_state.in_B, st.session_state.in_f)
    if exito:
        if st.session_state.step == 1:
            avanzar()
        st.rerun()

st.divider()

# ---------------------------------------------------------
# ETAPA 2: HOMOGENEIZACIÓN
# ---------------------------------------------------------
if st.session_state.step >= 2:
    data = st.session_state.math_data
    st.header("2. Homogeneizando las fronteras")
    
    st.markdown("Proponemos separar la solución en dos contribuciones físicas:")
    st.latex(rf"{COLOR_MAP['u']}(x,t) = {COLOR_MAP['w']}(x,t) + {COLOR_MAP['v']}(x,t)")
    
    st.markdown("Interpolando linealmente las fronteras, la **parte estacionaria** resulta en:")
    st.latex(rf"{COLOR_MAP['w']}(x,t) = {sp.latex(data['w'])}")
    
    st.markdown("Sustituyendo en la EDP, nuestro **nuevo problema transitorio** queda sujeto a:")
    
    with st.container(border=True):
        st.latex(rf"{COLOR_MAP['F_tilde']}(x,t) = {sp.latex(data['F_tilde'])}")
        st.latex(rf"{COLOR_MAP['f_tilde']}(x) = {sp.latex(data['f_tilde'])}")
        
    if st.session_state.step == 2:
        if st.button("Buscar Valores Propios 🚀"): avanzar(); st.rerun()
    st.divider()

# ---------------------------------------------------------
# ETAPA 3: VALORES Y FUNCIONES PROPIAS
# ---------------------------------------------------------
if st.session_state.step >= 3:
    data = st.session_state.math_data
    st.header("3. Valores y Funciones Propias Espaciales")
    
    st.markdown("Resolviendo el problema de Sturm-Liouville asociado a las fronteras homogéneas nulas en ambos extremos, obtenemos la base espacial:")
    
    with st.container(border=True):
        st.latex(rf"{COLOR_MAP['lam']}_n = {sp.latex(data['lam_n'])}")
        st.latex(rf"{COLOR_MAP['phi']}_n(x) = {sp.latex(data['phi_n'])}")

    if st.session_state.step == 3:
        if st.button("Plantear Expansión 🚀"): avanzar(); st.rerun()
    st.divider()

# ---------------------------------------------------------
# ETAPA 4: EXPANSIÓN Y STURM-LIOUVILLE
# ---------------------------------------------------------
if st.session_state.step >= 4:
    data = st.session_state.math_data
    st.header("4. Sturm-Liouville y Expansión")
    
    st.markdown("Expresamos la solución transitoria y la fuente modificada como combinaciones lineales (series de Fourier) de nuestras funciones propias espaciales:")
    
    L_tex = sp.latex(data['L'])
    sin_term = rf"\sin\left(\frac{{n\pi x}}{{{L_tex}}}\right)"
    
    with st.container(border=True):
        st.latex(rf"{COLOR_MAP['v']}(x,t) = \sum_{{n=1}}^{{\infty}} {COLOR_MAP['a']}_n(t) \cdot {sin_term}")
        st.latex(rf"{COLOR_MAP['F_tilde']}(x,t) = \sum_{{n=1}}^{{\infty}} {COLOR_MAP['q']}_n(t) \cdot {sin_term}")

    if st.session_state.step == 4:
        if st.button("Resolver Coeficientes Temporales 🚀"): avanzar(); st.rerun()
    st.divider()

# ---------------------------------------------------------
# ETAPA 5: COEFICIENTES TEMPORALES a_n(t)
# ---------------------------------------------------------
if st.session_state.step >= 5:
    data = st.session_state.math_data
    st.header("5. Obteniendo los Coeficientes Temporales")
    
    st.markdown("Proyectando la ecuación original sobre la base ortogonal, la EDP se desacopla en infinitas Ecuaciones Diferenciales Ordinarias (EDOs) para el tiempo:")
    
    factor = f"{sp.latex(data['alpha']**2)} \left(\\frac{{n\pi}}{{{sp.latex(data['L'])}}}\\right)^2"
    
    with st.container(border=True):
        st.latex(rf"{COLOR_MAP['a']}_n^\prime(t) + {factor} {COLOR_MAP['a']}_n(t) = {COLOR_MAP['q']}_n(t)")
        
    st.markdown("Donde los coeficientes de la fuente proyectada calculados por el producto interno son:")
    st.latex(rf"{COLOR_MAP['q']}_n(t) = {sp.latex(data['q_n_expr'])}")

    if st.session_state.step == 5:
        if st.button("Ensamblar Solución Final y Simular 🚀"): avanzar(); st.rerun()
    st.divider()

# ---------------------------------------------------------
# ETAPA 6: SOLUCIÓN Y SIMULADOR
# ---------------------------------------------------------
if st.session_state.step >= 6:
    data = st.session_state.math_data
    st.header("6. Solución General y Simulador Físico")
    
    vista = st.radio("Filtro de Visualización Matemática:", 
                     ["Solución General completa", "Solo Estacionaria", "Solo Transitoria"],
                     horizontal=True)
    
    L_tex = sp.latex(data['L'])
    w_tex = sp.latex(data['w'])
    sin_term = rf"\sin\left(\frac{{n\pi x}}{{{L_tex}}}\right)"
    sum_tex = rf"\sum_{{n=1}}^{{\infty}} {COLOR_MAP['a']}_n(t) \cdot {sin_term}"
    
    with st.container(border=True):
        if vista == "Solo Estacionaria":
            st.latex(rf"{COLOR_MAP['w']}(x,t) = {w_tex}")
        elif vista == "Solo Transitoria":
            st.latex(rf"{COLOR_MAP['v']}(x,t) = {sum_tex}")
        else:
            w_part = f"{w_tex} + " if data['w'] != 0 else ""
            st.latex(rf"{COLOR_MAP['u']}(x,t) = {w_part} {sum_tex}")

    # SIMULADOR NUMÉRICO
    st.subheader("Simulación Térmica Interactiva")
    t_val = st.slider(f"Flujo del Tiempo (t)", 0.0, 5.0, 0.0, 0.05)
    
    x_vals = np.linspace(0, data['L_num'], 250)
    u_vals = data['u_num_func'](x_vals, t_val)
    if np.isscalar(u_vals):
        u_vals = np.ones_like(x_vals) * u_vals

    fig, ax = plt.subplots(figsize=(10, 2), dpi=120)
    extent = [0, data['L_num'], 0, 1]
    
    im = ax.imshow(np.array([u_vals, u_vals]), cmap='inferno', aspect='auto', extent=extent, vmin=np.min(u_vals)-0.1, vmax=np.max(u_vals)+1)
    ax.set_yticks([])
    ax.set_xlabel('Posición de la barra $x$', fontweight='bold', color='#475569')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
    
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.3, pad=0.5)
    cbar.set_label(f'Temperatura a los {t_val:.2f}s', fontweight='bold', color='#475569')
    cbar.outline.set_visible(False)
    
    st.pyplot(fig)



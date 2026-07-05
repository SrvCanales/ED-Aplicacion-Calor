import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

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

# ==========================================
# DIÁLOGO DE AYUDA (Conceptos Separados)
# ==========================================
@st.dialog("📖 Profundización Matemática: Homogeneización", width="large")
def mostrar_ayuda_profunda(w_d, F_t_d, f_t_d):

    st.subheader("¿Por qué debemos homogeneizar?")

    st.markdown(r"""
El método de **separación de variables** funciona de manera natural cuando las
**condiciones de frontera son homogéneas**, es decir,

\[
u(0,t)=0,\qquad u(L,t)=0.
\]

Cuando en los extremos existen temperaturas distintas de cero, las autofunciones
ya no satisfacen directamente las condiciones de frontera y la expansión en series
resulta mucho más complicada.

La idea consiste en **restar únicamente la parte responsable de satisfacer las
condiciones de frontera**, dejando una nueva función que sí vale cero en los extremos.

La física del problema no cambia; únicamente cambiamos la forma en que describimos
la misma solución.
""")

    st.divider()

    st.subheader("La sustitución")

    st.markdown("""
Escribimos la solución como la suma de dos contribuciones:
""")

    st.latex(
        rf"{COLOR_MAP['u']}(x,t)="
        rf"{COLOR_MAP['w']}(x,t)+"
        rf"{COLOR_MAP['v']}(x,t)"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.success("### La parte $w(x,t)$")
        st.markdown(r"""
- Se construye **únicamente** para satisfacer las condiciones de frontera.

- Representa la parte que ya conocemos antes de resolver la EDP.

- Al escogerla correctamente, la nueva incógnita tendrá fronteras homogéneas.
""")

        st.latex(rf"{COLOR_MAP['w']}(x,t)={sp.latex(w_d)}")

    with col2:
        st.info("### La parte $v(x,t)$")
        st.markdown(r"""
Es la parte realmente desconocida.

Como

\[
u=w+v,
\]

entonces

\[
v=u-w.
\]

Por construcción,

\[
v(0,t)=v(L,t)=0.
\]

Ahora sí podremos desarrollar a \(v\) mediante autofunciones.
""")

    st.divider()

    st.subheader("¿Qué ocurre al sustituir?")

    st.markdown(r"""
Se reemplaza

\[
u=w+v
\]

en la ecuación diferencial y posteriormente se despeja la nueva incógnita \(v\).

Como consecuencia:

- las condiciones de frontera pasan a ser homogéneas;
- la condición inicial cambia;
- el término fuente también puede cambiar.

Todo esto ocurre automáticamente al efectuar la sustitución.
""")

    st.markdown("La nueva ecuación queda con:")

    st.latex(rf"{COLOR_MAP['F_tilde']}(x,t)={sp.latex(F_t_d)}")

    st.markdown("y la nueva condición inicial:")

    st.latex(
        rf"{COLOR_MAP['v']}(x,0)="
        rf"{sp.latex(f_t_d)}"
    )

    st.success(
        "A partir de este punto el problema posee fronteras homogéneas y ya puede "
        "resolverse mediante separación de variables."
    )

    st.divider()

    st.subheader("Preguntas frecuentes")

    ###########################################################################
    # ¿Por qué normalmente w es lineal?
    ###########################################################################

    if st.button(
        "❓ ¿Por qué normalmente se elige una función lineal?",
        use_container_width=True,
        key="faq_lineal"
    ):

        with st.expander("Respuesta", expanded=True):

            st.markdown(r"""
Cuando las temperaturas en los extremos son **constantes**, basta una recta que una
ambos valores.

Además,

\[
w''(x)=0,
\]

por lo que la sustitución introduce la menor cantidad posible de términos nuevos
en la ecuación diferencial.

Es la elección más sencilla y por ello aparece prácticamente en todos los cursos
introductorios.
""")

    ###########################################################################
    # ¿Siempre debe ser lineal?
    ###########################################################################

    if st.button(
        "❓ ¿Siempre debe ser una función lineal?",
        use_container_width=True,
        key="faq_no_lineal"
    ):

        with st.expander("Respuesta", expanded=True):

            st.markdown(r"""
No.

La única condición importante es que \(w\) satisfaga las condiciones de frontera.

Puede ser lineal, cuadrática, trigonométrica, exponencial o cualquier otra función
si ello simplifica el problema.
""")

    ###########################################################################
    # Casos donde NO es lineal
    ###########################################################################

    if st.button(
        "❓ Ejemplos donde la sustitución NO es lineal",
        use_container_width=True,
        key="faq_ejemplos"
    ):

        with st.expander("Ejemplos", expanded=True):

            st.markdown(r"""
### Ejemplo 1: frontera dependiente del tiempo

Si

\[
u(0,t)=10e^{-t},
\qquad
u(L,t)=40e^{-t},
\]

es natural escoger

\[
w(x,t)=
10e^{-t}
+\frac{30e^{-t}}{L}x.
\]

Aunque sigue siendo lineal en \(x\), **ya no es una función lineal completa**
porque depende del tiempo.

---

### Ejemplo 2: frontera senoidal

Si

\[
u(0,t)=5\sin t,
\qquad
u(L,t)=0,
\]

podemos usar

\[
w(x,t)=
5\sin t
\left(1-\frac{x}{L}\right).
\]

---

### Ejemplo 3: cuando queremos eliminar un término fuente

Supongamos

\[
u_t=\alpha^2u_{xx}+20.
\]

Puede ser conveniente elegir una función cuadrática

\[
w(x)=Ax^2+Bx+C,
\]

porque

\[
w''(x)=2A,
\]

permitiendo cancelar el término constante de la fuente.

---

### Ejemplo 4: estado estacionario conocido

Si sabemos que el estado estacionario satisface

\[
w''+\sin(x)=0,
\]

conviene utilizar directamente esa solución como \(w\).

Así, el término fuente desaparece completamente del problema para \(v\).

---

En resumen, **no existe una única sustitución correcta**.

Se escoge aquella que haga el problema lo más sencillo posible.
""")

    ###########################################################################
    # ¿Cómo sé cuál escoger?
    ###########################################################################

    if st.button(
        "❓ ¿Cómo saber qué función elegir?",
        use_container_width=True,
        key="faq_como"
    ):

        with st.expander("Respuesta", expanded=True):

            st.markdown(r"""
En la práctica suele seguirse este orden:

1. Buscar una función que satisfaga exactamente las condiciones de frontera.

2. Preferir la función más simple posible.

3. Si además simplifica la ecuación diferencial, mejor aún.

En cursos básicos, casi siempre una función lineal es suficiente.

En problemas más avanzados puede elegirse otra función para cancelar términos de
la EDP o aprovechar información física del problema.
""")

    ###########################################################################
    # ¿La solución cambia?
    ###########################################################################

    if st.button(
        "❓ ¿La solución física cambia después de la sustitución?",
        use_container_width=True,
        key="faq_fisica"
    ):

        with st.expander("Respuesta", expanded=True):

            st.markdown(r"""
No.

La temperatura física sigue siendo

\[
u=w+v.
\]

Nunca se reemplaza una solución por otra.

Simplemente se resuelve primero la parte transitoria \(v\), porque es mucho más
fácil, y al final se vuelve a sumar \(w\).

Por ello, la homogeneización es únicamente un cambio de representación matemática,
no un cambio del fenómeno físico.
""")

    ###########################################################################
    # ¿Por qué cambia la condición inicial?
    ###########################################################################

    if st.button(
        "❓ ¿Por qué también cambia la condición inicial?",
        use_container_width=True,
        key="faq_ci"
    ):

        with st.expander("Respuesta", expanded=True):

            st.markdown(r"""
Al inicio,

\[
u(x,0)=f(x).
\]

Como

\[
u=w+v,
\]

entonces necesariamente

\[
v(x,0)=f(x)-w(x,0).
\]

Es decir, la nueva condición inicial representa únicamente la parte que aún falta
por evolucionar después de haber retirado el perfil \(w\).
""")

    st.divider()

    st.info(
        "💡 **Idea clave:** La homogeneización consiste únicamente en separar de la "
        "solución una parte conocida (w), de modo que la parte restante (v) tenga "
        "condiciones de frontera homogéneas y pueda resolverse mediante separación "
        "de variables. Al finalizar el proceso, ambas contribuciones se vuelven a "
        "sumar para recuperar la temperatura física original."
    )

# ==========================================
# INTERFAZ PRINCIPAL - ETAPA 1
# ==========================================
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

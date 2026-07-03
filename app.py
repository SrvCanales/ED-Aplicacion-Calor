import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Símbolos matemáticos
x, t, tau = sp.symbols('x t tau')
n_sym = sp.Symbol('n', integer=True, positive=True)

st.set_page_config(page_title="Simulador de Calor 1D", layout="wide")
st.title("🔥 Simulador de EDP: Calor 1D")

# BARRA LATERAL: El estudiante ingresa todo aquí de forma cómoda
st.sidebar.header("Configuración del PVIF")
L_input = st.sidebar.text_input("Longitud L:", value="pi")
alpha_input = st.sidebar.text_input("Difusividad α:", value="1")
F_input = st.sidebar.text_input("Fuente F(x,t):", value="exp(-t)*sin(2*x)")
A_input = st.sidebar.text_input("Frontera Izquierda u(0,t):", value="0")
B_input = st.sidebar.text_input("Frontera Derecha u(L,t):", value="0")
f_input = st.sidebar.text_input("Condición Inicial u(x,0):", value="3*sin(4*pi*x)")
N_terms = st.sidebar.slider("Términos para la simulación numérica (N):", 1, 15, 5)

# Pestañas para simular la "ruta guiada" por pasos sin ensuciar la pantalla
tab1, tab2, tab3 = st.tabs(["📘 1. Homogenización y Autofunciones", "📐 2. Coeficientes Temporales", "📊 3. Simulador Interactivo"])

# Procesamiento matemático global (SymPy)
try:
    L = sp.sympify(L_input, locals={'pi': sp.pi})
    alpha = sp.sympify(alpha_input)
    F = sp.sympify(F_input, locals={'x': x, 't': t, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp})
    A = sp.sympify(A_input)
    B = sp.sympify(B_input)
    f = sp.sympify(f_input, locals={'x': x, 'pi': sp.pi, 'sin': sp.sin})
    
    w = sp.simplify(A + (x / L) * (B - A))
    F_tilde = sp.simplify(F - sp.diff(w, t) + alpha**2 * sp.diff(w, x, 2))
    f_tilde = sp.simplify(f - w.subs(t, 0))

    # --- PESTAÑA 1: HOMOGENIZACIÓN ---
    with tab1:
        st.markdown("### 1.2 Homogeneizando las fronteras")
        st.write("Para resolver mediante separación de variables, eliminamos las fronteras no homogéneas dividiendo el problema:")
        st.latex(fr"w(x,t) = {sp.latex(w)}")
        st.write("Esto nos genera el sistema transitorio con una fuente modificada:")
        st.latex(fr"\tilde{{F}}(x,t) = {sp.latex(F_tilde)}")
        
        st.markdown("### 1.3 Valores y funciones propias espaciales")
        lam_n = n_sym * sp.pi / L
        st.latex(fr"\lambda_n = \left(\frac{{n\pi}}{{{sp.latex(L)}}}\right)^2 \quad ; \quad \phi_n(x) = \sin\left(\frac{{n\pi x}}{{{sp.latex(L)}}}\right)")

    # --- PESTAÑA 2: COEFICIENTES ---
    with tab2:
        st.markdown("### 1.5 Coeficientes Temporales y Solución General")
        st.write("La EDP parcial se transforma en un conjunto infinito de EDOs ordinarias para cada componente temporal $T_n(t)$:")
        st.latex(fr"T_n'(t) + {sp.latex(alpha**2)}\lambda_n^2 T_n(t) = q_n(t)")
        
        # Mostrar la estructura en papel solicitada
        u_inf_tex = (sp.latex(w) + r" + ") if w != 0 else ""
        u_inf_tex += r"\sum_{n=1}^{\infty} T_n(t) \sin\left(\frac{n\pi x}{" + sp.latex(L) + r"}\right)"
        st.markdown("#### Expresión Algebraica Formal:")
        st.latex(fr"u(x,t) = {u_inf_tex}")

    # --- PESTAÑA 3: SIMULADOR ---
    with tab3:
        st.markdown("### 2.1 Distribución Térmica en Tiempo Real")
        
        # Control del tiempo interactivo
        t_val = st.slider("Desliza para avanzar el tiempo (t)", 0.0, 5.0, 0.0, 0.05)
        
        # Cálculo de la solución truncada para la gráfica
        v_sol = 0
        for n in range(1, N_terms + 1):
            lam_val = n * sp.pi / L
            phi_val = sp.sin(lam_val * x)
            q_n = (2/L) * sp.integrate(F_tilde * phi_val, (x, 0, L))
            T_n_0 = (2/L) * sp.integrate(f_tilde * phi_val, (x, 0, L))
            factor_k = alpha**2 * lam_val**2
            int_part = sp.integrate(q_n.subs(t, tau) * sp.exp(factor_k * tau), tau)
            int_eval = int_part.subs(tau, t) - int_part.subs(tau, 0)
            T_n = sp.exp(-factor_k * t) * (int_eval + T_n_0)
            v_sol += T_n * phi_val

        u_final_num = sp.simplify(w + v_sol)
        L_num_val = float(L.evalf())
        u_num_func = sp.lambdify((x, t), u_final_num, modules=['numpy', 'math'])
        
        # Graficación con Matplotlib
        x_vals = np.linspace(0, L_num_val, 200)
        u_vals = u_num_func(x_vals, t_val)
        if np.isscalar(u_vals):
            u_vals = np.ones_like(x_vals) * u_vals
            
        fig, ax = plt.subplots(figsize=(10, 2))
        im = ax.imshow(np.array([u_vals, u_vals]), cmap='inferno', aspect='auto', extent=[0, L_num_val, 0, 1])
        ax.set_yticks([])
        ax.set_xlabel('Posición de la barra x')
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.3, pad=0.5)
        
        st.pyplot(fig)

except Exception as e:
    st.error(f"Error en el procesamiento matemático. Asegúrate de escribir expresiones válidas. Detalle: {e}")
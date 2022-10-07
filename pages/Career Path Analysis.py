#------------------- Input library -------------------
import pandas as pd
import streamlit as st
import transformations as tr
import io
import distutils
#------------------- Visualization Settings -------------------
skill_color_list = ['rgba(8,17,129,.8)','rgba(38,19,142,.8)','rgba(63,21,153,.8)',
              'rgba(81,24,157,.8)','rgba(99,26,161,.8)','rgba(118,32,158,.8)',
              'rgba(126,39,155,.8)','rgba(152,48,146,.8)','rgba(166,59,138,.8)',
              'rgba(179,72,127,.8)','rgba(191,84,117,.8)','rgba(203,98,107,.8)',
              'rgba(216,119,97,.8)','rgba(223,126,92,.8)','rgba(229,142,86,.8)',
              'rgba(235,155,82,.8)','rgba(241,177,79,.8)','rgba(244,195,78,.8)',
              'rgba(243,215,81,.8)','rgba(243,235,85,.8)']

st.session_state['plot_radius'] =  60
rad_adjust = 90

st.session_state['df'] = pd.DataFrame()
st.session_state['df_roles'] = pd.DataFrame()

st.session_state['total_roles'] = st.slider("Number of positions", 1, 10)
st.session_state['blank_roles'] = ['Position'+ ' ' +(str(x+1)) for x in range(0,st.session_state.total_roles)]
st.session_state['blank_dict'] = {}
for x in st.session_state.blank_roles:
        st.session_state['blank_dict'][x] = {'Role_name':'blank','Years':0,'Skills':[]}
    
for i,x in enumerate(st.session_state['blank_roles']):
        #blank_dict = persistdata()
        with st.expander(label = x, expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                #Name of role
                st.session_state.blank_dict[x]['Role_name'] = st.text_input(
                    label = 'Name of position',
                    placeholder=x,
                    max_chars=25,
                    key="blank"+str(i+1))
                
            with col2:
                st.session_state.blank_dict[x]['Years'] = st.slider("Years in position", min_value=0.5, max_value=15.0, value=1.0, step=.25, key=x+'_'+str(i))
            st.session_state.blank_dict[x]['Skills'].append(st.text_input(label = 'Top skill 1',placeholder='Most used skill in this position', max_chars=25, key=x+'_'+'skill1'))
            st.session_state.blank_dict[x]['Skills'].append(st.text_input(label = 'Top skill 2',placeholder='Next most used skill or leave blank', max_chars=25, key=x+'_'+'skill2'))
            st.session_state.blank_dict[x]['Skills'].append(st.text_input(label = 'Top skill 3',placeholder='Next most used skill or leave blank', max_chars=25, key=x+'_'+'skill3'))
            st.session_state.blank_dict[x]['Skills'].append(st.text_input(label = 'Top skill 4',placeholder='Next most used skill or leave blank', max_chars=25, key=x+'_'+'skill4'))
for x in st.session_state.blank_dict:
    temp = (pd.DataFrame(index=[x for x in st.session_state.blank_dict[x]['Skills'] if x !=""],
                         columns=[st.session_state.blank_dict[x]['Role_name']],
                         data=st.session_state.blank_dict[x]["Years"]))
    st.session_state.df = pd.concat([st.session_state.df,temp],axis=1)
    st.session_state.df_roles = st.session_state.df.T.max(axis=1).to_frame().reset_index()
    st.session_state.df_roles.columns = ['Role','Years']
try:
    
    st.session_state.df = st.session_state.df.fillna(0.0)
    st.session_state.df['Years'] = st.session_state.df.sum(axis=1)
    st.session_state.df = st.session_state.df[st.session_state.df['Years'] != 0]
    st.session_state.df = st.session_state.df.sort_values('Years', ascending=False)
    st.session_state.df_roles = st.session_state.df_roles.fillna(0.0)
    st.session_state['skill_role_dict'] = tr.create_skill_role_dict(df=st.session_state.df)
    st.session_state['skill_dict'], st.session_state['skill_intervals'] = tr.skill_plot_data(df=st.session_state.df)
    #Controls for graph
    cola, colb, colc = st.columns(3)
    with cola:
        st.session_state['marker_increase'] = st.slider("Marker size", 1, 8, 4,key='marker_size')
        
    with colb:
        st.session_state['text_size'] = st.slider("Text size", 10, 20, 15,key='text-size')

    with colc:
        st.session_state['line_width'] = st.slider("Line width", 1, 8, 1,)
    
    cold, cole = st.columns(2)
    with cold:
        st.session_state['chartsize'] = st.slider("Chart size", 400, 800, 600,step=100,key='chart_size')

    with cole:
        st.session_state['showlegend'] = bool(distutils.util.strtobool(st.radio(
        "Show skill legend",
        ('False', 'True'),horizontal=True)))
    
    st.session_state['role_dict'] = tr.create_role_dict(
        df_roles=st.session_state.df_roles, 
        plot_radius=st.session_state.plot_radius
        )

    st.session_state['fig'] = tr.create_polar_chart(
        df = st.session_state.df, 
        df_roles = st.session_state.df_roles, 
        skill_role_dict = st.session_state.skill_role_dict, 
        skill_intervals = st.session_state.skill_intervals, 
        role_dict = st.session_state.role_dict, 
        skill_dict = st.session_state.skill_dict,
        skill_color_list = skill_color_list,
        marker_increase = st.session_state.marker_increase, 
        rad_adjust = rad_adjust,
        plot_radius=st.session_state.plot_radius,
        text_size=st.session_state.text_size,
        line_width = st.session_state.line_width,
        chartsize=st.session_state.chartsize,
        showlegend=st.session_state.showlegend
        )
    st.plotly_chart(st.session_state.fig)
    
    except Exception as e:
    st.write("Your visualization will apprear here when you've entered enough data.")

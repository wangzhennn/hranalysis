#from turtle import position
import numpy as np
import textwrap as tw
import plotly.graph_objects as go
import pandas as pd



def customwrap(s,width=8):
    return "<br>".join(tw.wrap(s,width=width,break_long_words=False))

def create_skill_role_dict(df):    
    # Dictionary with roles for every skill: skills = keys and roles = values
    skill_role_dict = {}
    for i in df.index:
        temp = df.copy()
        if isinstance(temp, pd.DataFrame):
            temp = temp.loc[i]
            temp = temp[temp>0].dropna()[:-1]
            skill_role_dict[i] = list(temp.index)
        else:
            temp = temp.loc[i].to_frame()
            temp = temp[temp>0].dropna()[:-1]
            skill_role_dict[i] = list(temp.index)
    
    return skill_role_dict

    #Data prep for plotting skills
def skill_plot_data(df):
    #Number of skills
    skill_len = len(df)

    #Max years used as radius of circle
    max_years = int(np.round(df['Years'],0)[0])

    #Must have 4-20 skills
    min_max_skills = [x for x in range(1,21)]

    #Defining radial marks for each set of skills
    skill_temp = [360/x for x in min_max_skills]
    skill_inc_dict = dict(zip(min_max_skills,skill_temp))
    skill_intervals = [x for x in np.arange(0,360,skill_inc_dict[skill_len])]
    skill_intervals.reverse()
    skill_dict = dict(zip(df.index,skill_intervals))

    return skill_dict, skill_intervals

def create_role_dict(df_roles,plot_radius):    
    #Data prep for plotting roles
    roles = len(df_roles)
    roles_radius = plot_radius*.75
    roles_diameter = roles_radius*2

    if (roles % 2) == 0:
        #roles_plots = [x for x in np.arange(min(df_roles['Years']),roles_radius,roles/2)]
        cuttoff = int(roles/2)
        
        step = roles_diameter/roles
        
        
        start_direction = list([180]*cuttoff)
        start_plots = [x for x in np.arange(0+step/2,roles_radius,step)]
        start_plots.reverse()
        
        
        end_direction = list([0] * cuttoff)
        end_plots = start_plots.copy()
        end_plots.reverse()
        
        plots = start_plots+end_plots
        directions = start_direction+end_direction
        

    else:
        cuttoff = int(np.floor(roles/2))
        
        step = roles_diameter/roles

        start_roles = df_roles['Role'].tolist()[:cuttoff]
        start_direction = list([180]*cuttoff)
        start_plots = [x for x in np.arange(0+step,roles_radius,step)]
        start_plots.reverse()

        middle_role = df_roles['Role'].tolist()[cuttoff]
        middle_direction = [0]
        middle_plots = [0]

        end_roles = df_roles['Role'].tolist()[cuttoff+1:]
        end_direction = list([0] * cuttoff)
        end_plots = start_plots.copy()
        end_plots.reverse()

        plots = start_plots+middle_plots+end_plots
        directions = start_direction+middle_direction+end_direction

    df_roles['Plots']=plots
    df_roles['Direction']=directions

    role_dict = dict(zip(list(df_roles['Role']),zip(plots, directions)))
    
    return role_dict

def skill_in_role_line(role_dict,skill_dict,skill,role,plot_radius,rad_adjust):
    role_plot = role_dict[role][0]
    direction = role_dict[role][1]
    skill_plot = skill_dict[skill]

    r = [role_plot,plot_radius] #x
    theta = [direction,skill_plot-rad_adjust] #y

    r_line = list(np.linspace(r[0], r[1],50))
    theta_line = list(np.linspace(theta[0], theta[1]*1,50))

    return r_line, theta_line

#def skill_line(role_plot,direction, skill_plot,plot_radius, rad_adjust):
 #   r = [role_plot,plot_radius] #x
  #  theta = [direction,skill_plot-rad_adjust] #y

   # r_line = list(np.linspace(r[0], r[1],50))
    #theta_line = list(np.linspace(theta[0], theta[1]*1,50))
    
    #return r_line, theta_line

def create_polar_chart(
    df, 
    df_roles, 
    skill_role_dict,
    role_dict,
    skill_dict, 
    skill_intervals, 
    skill_color_list,
    marker_increase, 
    rad_adjust, 
    plot_radius,
    text_size,
    line_width,
    chartsize,
    showlegend=False):
    #Create figure
    fig = go.Figure()
    
    # Plotting lines from skill plot positions to each related role
    for i in skill_role_dict:
        for x in skill_role_dict[i]:
            r_line, theta_line = skill_in_role_line(
                role_dict=role_dict,
                skill_dict=skill_dict,
                role=x,
                skill=i,
                plot_radius=plot_radius,
                rad_adjust=rad_adjust) 

            fig.add_trace(go.Scatterpolar(
                name = x+''+i+''+'line',
                r = r_line, 
                theta = theta_line, #y
                mode = 'lines',
                line_color = 'rgba(244,195,78,.8)',
                line={'width':line_width},
                hoverinfo='skip',
                showlegend=False
        ))

    # Plotting skills in circle path with white background
    for index,x in enumerate(skill_intervals):
        marker_name = df.index[index]
        marker_size = int(np.round(df['Years'][index]))
        fig.add_trace(
            go.Scatterpolar(
                name = marker_name+''+'white',
                r = [plot_radius],
                theta = [x-rad_adjust],
                mode = 'markers',
                marker_color='white',
                marker_size=marker_size*marker_increase*1,
                hoverinfo='skip',
                showlegend=False

        ))

    # Plotting skills in circle path with colors    
    for index,x in enumerate(skill_intervals):
        marker_name = df.index[index]
        marker_size_float = df['Years'][index]
        marker_size = int(np.round(df['Years'][index]))
        text_pos = ['middle right', 'top right', 'top left', 'middle left', 'bottom left', 'bottom right']
        if x-rad_adjust == 0:
            pos = text_pos[0]
        if 0 < x-rad_adjust < 90:
            pos = text_pos[1]
        if 90 <= x-rad_adjust < 180: 
            pos = text_pos[2]
        if x-rad_adjust == 180:
            pos = text_pos[3]
        if 180 < x-rad_adjust < 270:
            pos = text_pos[4]
        if 270 <= x-rad_adjust < 366: 
            pos = text_pos[5]
        
        fig.add_trace(
            go.Scatterpolar(
                name = marker_name+': '+str(marker_size_float)+' '+'years',
                r = [plot_radius],
                theta = [x-rad_adjust],
                mode = 'markers+text',
                marker_color=skill_color_list[index],
                marker_size=marker_size*marker_increase,
                text=customwrap(marker_name),
                textposition=pos,
                textfont = (
                    {'color':'rgba(108, 122, 137,7)',
                    'size':text_size
                    }
                ),
                hovertemplate = '{}: {} years<extra></extra>'.format(marker_name, marker_size),
                showlegend=showlegend
        ))

    # Plotting roles in a line with white background
    for index,x in enumerate(df_roles['Plots']):
        marker_name = df_roles['Role'][index]
        marker_size = int(np.round(df_roles['Years'][index]))
        fig.add_trace(
            go.Scatterpolar(
                name = marker_name+''+'white',
                r = [x],
                theta = [df_roles['Direction'][index]],
                mode = 'markers',
                marker_color='white',
                marker_size=marker_size*marker_increase*1,
                hoverinfo='skip',
                showlegend=False
        ))

    # Plotting roles in a line with colors
    text_num = 1
    text_pos = ['top center', 'bottom center']
    for index,x in enumerate(df_roles['Plots']):
        if text_num %2:
            pos = text_pos[0]
        else:
            pos = text_pos[1]
        marker_name = df_roles['Role'][index]
        marker_size = int(np.round(df_roles['Years'][index]))
        fig.add_trace(
            go.Scatterpolar(
                name= marker_name+''+'color',
                r = [x],
                theta = [df_roles['Direction'][index]],
                mode = 'text+markers',
                text=customwrap(marker_name),
                textposition=pos,
                textfont = (
                    {'color':'rgba(108, 122, 137,7)',
                    'size':text_size
                    }
                ),
                #marker_color=job_color_list[df_roles['Rank'][index]],
                marker_color='turquoise',
                marker_size=marker_size*marker_increase*1.75,
                hovertemplate = '{}: {} years<extra></extra>'.format(marker_name, marker_size),
                showlegend=False
            ))
        text_num = text_num+1


    fig.update_layout(
        title='Your non-linear career',
        titlefont =(
                    {'color':'rgba(108, 122, 137,1)',
                    'size':18
                    }
                ),
        width=chartsize,
        height=chartsize,
        #margin = {'l': 300},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        #autosize=True,
        template=None,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=-0.9,
        ),
        polar = dict(
            bgcolor = "rgba(0, 0, 0,0)",
            angularaxis = dict(
                gridcolor='rgba(223,223,233,0)',
                linewidth = 3,
                showline=False,
                linecolor='rgba(0,223,233,0)',
                showticklabels=False,
                tickcolor='rgba(223,223,233,0)',
            ),
            radialaxis = dict(
                side = "counterclockwise",
                showline = False,
                linewidth = 2,
                gridcolor = 'rgba(223,223,233,0)',
                gridwidth = 2,
                showticklabels=False,
                tickcolor='rgba(223,223, 223, 0)'
            )
        ))
    
    return fig

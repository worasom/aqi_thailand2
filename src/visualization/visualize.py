# -*- coding: utf-8 -*-
from ..imports import *
from ..gen_functions import *

def plot_dendogram(data:pd.core.frame.DataFrame, cols=None, front_size=16,filename=None):
    """Plot hierarchical relationship between features in the data. 
    
    Args:
        data: dataframe from dataset.data
        cols(optional): columns to use for making the dendogram. If None, use the columns of the data dataFrame  
        front_size(optional): plot front size [default:None]
        filename(optional): if not None, save plot as filename[default:None]

    """
    if cols is None:
        cols = data.columns
    # Redundant Features
    corr = np.nan_to_num(np.round(spearmanr(data[cols]).correlation, 4))

    for i in range(corr.shape[0]):
        corr[i, i] = 1

    corr_condensed = hc.distance.squareform(1 - corr)
    z = hc.linkage(corr, method='average')
    _, ax = plt.subplots(figsize=(16, 10))
    dendrogram = hc.dendrogram(
        z,
        labels=cols,
        orientation='left',
        leaf_font_size=front_size)
    
    if filename:
        plt.savefig(filename)


def display_time_split(index_list):
    """ Display how the time series data is split.

    Args: 
        index_list: a list of index in each split 

    """
    lenght = len(index_list)
    if lenght==3:
        colors = ['royalblue','orange','red']
        label_list = ['Training', 'Validation', 'Test']
    
    elif lenght == 4:
        colors = ['royalblue','orange','red','maroon']
        label_list = ['Training', 'Validation1', 'Validation2', 'Test']
    
    else:
        colors = get_color(color_length= length, cmap=cm.jet)

    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax_list = []
    for idxs, color in zip(index_list,colors):
        # Plot training and test indices
        l1 = ax.scatter(idxs, [1] * len(idxs), c=color, marker='_', lw=6)
        ax_list.append(l1)
     
    ax.legend(ax_list, label_list)
    ax.set(ylim=[0.8, 1.5], title='Split behavior', xlabel='date')
    plt.xticks(rotation=90)


def show_fea_imp(fea_imp,x_log=False, filename=None,title=''):
    """Display feature of importance in a bar plot 

    Args: 
        imp_df: important dataframe 
        x_log: if True, plot x axis in a long scale
        filename: filename to save figure as 
        title: figure title
    
    """
    plt.rcParams.update({'font.size': 14})


    if 'imp_std' in fea_imp.columns:
        fea_imp.drop(0).plot('index','importance',kind='barh',xerr='imp_std',figsize=(5,8),linewidth=1,edgecolor='black',legend=False)
    else:
        fea_imp.drop(0).plot('index','importance',kind='barh',figsize=(5,8),linewidth=1,edgecolor='black', legend=False)

    if x_log:
        plt.xscale('log')

    plt.title(title)
    plt.xlabel('importance index')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)


def plot_corr(poll_df, avg='d',filename=None):
    """Plot the correlation between different pollutants 
    
    Args:
        poll_df:
        avg: 'd' or 'm' 

    """
    df = poll_df.resample(avg).max()
    plt.figure(figsize=(10,8))
    mask = np.tril(df.corr())
    # use spearman rank correlation 
    sns.heatmap(df.corr(method='spearman'), annot=True,mask=mask)
    
    if filename:
        plt.savefig(filename)

def plot_season_avg(poll_df, pollutant, ax, plot_error=True):
    """Plot the average by date of year. Good for looking seasonal pattern.

    """
    plt.rcParams.update({'font.size': 14})

    df = poll_df[[pollutant]].resample('d').max().copy().dropna()
    df['dayofyear'] = df.index.dayofyear
    df['year'] = df.index.year

    # add winter day by substratcing the first day of july
    winterday = df['dayofyear'] - 182
    # get rid of the negative number
    winter_day_max = winterday.max()
    winterday[winterday < 0] = winterday[winterday < 0] + 182 + winter_day_max  
    df['winter_day'] = winterday

    # add month-day 
    df['month_day'] = df.index.strftime('%m-%d')
    temp = df[['winter_day', 'month_day']].set_index('winter_day')
    temp.index = temp.index.astype(str)
    winter_day_dict = temp.to_dict()['month_day']

    if plot_error:
        sns.lineplot(data=df,x='winter_day', y=pollutant, ax=ax, legend='brief',label=pollutant,color='blue')
    
    else:
        mean_day = df.groupby('winter_day').mean()[pollutant]
        ax.plot(mean_day,label=pollutant)

    ax.set_xlim([0, 366])
    new_ticks = ['07-01', '08-20', '10-09', '11-28', '01-16', '03-06', '04-25', '06-14', '']         
        
    ax.set_xticklabels(new_ticks)
    ax.set_xlim([df['winter_day'].min(), df['winter_day'].max()])
    ax.legend()
    ax.set_xlabel('month-date')
    #plt.show()
    return winter_day_dict



def plot_all_pollutions(poll_df, city_name='',filename=None,transition_dict=None, color_labels=None, level_name=None):
    """Plot all pollutant data over time. 

    Args:
        poll_df  
        city_name 
        filename=None
        transition_dict=None
        color_labels=None 
        level_name=None
    
    """
    if transition_dict==None:
        transition_dict = { 'PM2.5': [0, 35.5, 55.4, 150.4, 1e3],
            'PM10': [0, 155, 254, 354, 1e3],
            'O3':[0, 70 , 85, 105 ,1e3],
            'SO2':[0, 75, 185, 304,1e3],
            'NO2': [0, 100, 360, 649,1e3],
            'CO': [0, 6.4, 12.5, 15.4,1e3]}

    if color_labels==None:
        color_labels = ['green', 'orange', 'red','purple']
    
    if level_name==None:
        level_names = ['satisfactory', 'moderate', 'unhealthy','very unhealthy']
    
    gas_list = poll_df.columns
    len_gas = len(gas_list)
    _, ax = plt.subplots(len_gas, 1, figsize=(10, 3*len_gas),sharex=True)
    

    for i, _ in enumerate(ax):
        col = gas_list[i]
        poll_data = poll_df[[col]].dropna()
        poll_data['color'] = pd.cut(poll_data[col], bins=transition_dict[col], labels=color_labels)

        for color, legend in zip(color_labels, level_names):
            temp = poll_data[poll_data['color'] == color]
            # plot the data for each pollution level
            ax[i].scatter(temp.index, temp[col], c=temp['color'], marker='.', label=legend,alpha=0.7)
        
            if col in ['PM10','PM2.5']:
                ax[i].set_ylabel(col + '($\mu g/m^3$)')
            elif col in ['O3','NO2','SO2']:
                ax[i].set_ylabel(col + '(ppb)')
            elif col == 'CO':
                ax[i].set_ylabel(col + '(ppm)')
        
            ax[i].axhline(transition_dict[col][1],color='orange')
            if i in [0,1,2]:
                ax[i].axhline(transition_dict[col][2],color='red')
            if i in [0,1,2]:
                ax[i].axhline(transition_dict[col][3],color='purple')
    
    ax[0].legend(loc='upper left')
    ax[0].set_xlim([poll_df.index.min(), poll_df.index.max()])
    ax[0].set_title(f'Pollutants Data for {city_name} Averaged from All Staions')
    ax[-1].set_xlabel('date')

    plt.tight_layout()

    if filename:
        plt.savefig(filename)

def compare_us_thai_aqi():
    """Plot the different between US and Thailand AQI conversion.

    """
    plt.rcParams.update({'font.size': 14})
    _, ax = plt.subplots(6,1, figsize=(12, 15))

    ax[0].set_title('PM2.5(24hr avg) AQI converstion')
    # Thailand 
    ax[0].plot(np.arange(0, 25), np.zeros(len(np.arange(0, 25))),c='royalblue', marker='_', lw=18, label='very good')



    # us
    ax[0].plot(np.arange(0, 12), np.ones(len(np.arange(0, 12))),
        c='green', marker='_', lw=18, label='good/satisfactory')
    ax[0].plot(np.arange(12, 35.5), np.ones(len(np.arange(12, 35.5))),
        c='yellow', marker='_', lw=18, label='moderate')
    ax[0].plot(np.arange(35.5, 55.5), np.ones(len(np.arange(35.5, 55.5))),
        c='orange', marker='_', lw=18, label='unhealthy for \n sensitive group')
    ax[0].plot(np.arange(55.5, 180.5), np.ones(len(np.arange(55.5, 180.5))),
        c='red', marker='_', lw=18, label='unhealthy')
    ax[0].plot(np.arange(180.5, 200.0), np.ones(len(np.arange(180.5, 200))),
        c='purple', marker='_', lw=18, label='very unhealthy')


    # the rest of Thailand
    ax[0].plot(np.arange(25, 37), np.zeros(
    len(np.arange(25, 37))), c='green', marker='_', lw=18)
    ax[0].plot(np.arange(37, 50), np.zeros(len(np.arange(37, 50))),
        c='yellow', marker='_', lw=18)
    ax[0].plot(np.arange(50, 91), np.zeros(len(np.arange(50, 91))),
        c='orange', marker='_', lw=18)
    ax[0].plot(np.arange(91, 200), np.zeros(
    len(np.arange(91, 200))), c='red', marker='_', lw=18)
    ax[0].set_xlabel('PM2.5($\mu$g/m$^3$)')

    #--AQI label--(TH)
    ax[0].text(13,-0.2,'25')
    ax[0].text(26,-0.2,'50') 
    ax[0].text(35,-0.2,'100')
    ax[0].text(75,-0.2,'200')
    #ax[0].text(190,-0.2,'310')

    #--AQI label--(US)
    ax[0].text(0,0.8,'50') 
    ax[0].text(22,0.8,'100')
    ax[0].text(40,0.8,'150')
    ax[0].text(165,0.8,'200')
    #ax[0].text(192,0.8,'300')

    #-----------------------PM10---------------------------

    ax[1].set_title('PM10(24hr avg) AQI converstion')

    #-------------us-----------------
    ax[1].plot(np.arange(0, 55), np.ones(len(np.arange(0, 55))),
        c='green', marker='_', lw=18, label='good/satisfactory')
    ax[1].plot(np.arange(55, 155), np.ones(len(np.arange(55, 155))),
        c='yellow', marker='_', lw=18, label='moderate')
    ax[1].plot(np.arange(155, 255), np.ones(len(np.arange(155, 255))),
        c='orange', marker='_', lw=18, label='unhealthy for \n sensitive group')
    ax[1].plot(np.arange(255, 355), np.ones(len(np.arange(255, 355))),
        c='red', marker='_', lw=18, label='unhealthy')
    ax[1].plot(np.arange(355, 425), np.ones(len(np.arange(355, 425))),
        c='purple', marker='_', lw=18, label='very unhealthy')

    #-------------Thailand -------------------
    ax[1].plot(np.arange(0, 50), np.zeros(len(np.arange(0, 50))),
        c='royalblue', marker='_', lw=18, label='very good')
    ax[1].plot(np.arange(50, 80), np.zeros(
    len(np.arange(50, 80))), c='green', marker='_', lw=18)
    ax[1].plot(np.arange(80, 120), np.zeros(len(np.arange(80, 120))),
        c='yellow', marker='_', lw=18)
    ax[1].plot(np.arange(120, 180), np.zeros(len(np.arange(120, 180))),
        c='orange', marker='_', lw=18)
    ax[1].plot(np.arange(180, 425), np.zeros(
    len(np.arange(180, 425))), c='red', marker='_', lw=18)
    ax[1].set_xlabel('PM10($\mu$g/m$^3$)')

    #--AQI label--(TH)
    ax[1].text(25,-0.2,'25')
    ax[1].text(58,-0.2,'50') 
    ax[1].text(90,-0.2,'100')
    ax[1].text(150,-0.2,'200')
    #ax[0].text(400,-0.2,'323')

    #--AQI label--(US)
    ax[1].text(25,0.8,'50') 
    ax[1].text(125,0.8,'100')
    ax[1].text(225,0.8,'150')
    ax[1].text(325,0.8,'200')
    #ax[1].text(405,0.8,'300')

    #-----------------------O3---------------------------

    ax[2].set_title('O$_3$(8hr avg) AQI converstion')


    ax[2].plot(np.arange(0, 54), np.ones(len(np.arange(0, 54))),
        c='green', marker='_', lw=18, label='good/satisfactory')
    ax[2].plot(np.arange(54, 70), np.ones(len(np.arange(54,  70))),
        c='yellow', marker='_', lw=18, label='moderate')
    ax[2].plot(np.arange(70, 85), np.ones(len(np.arange(70, 85))),
        c='orange', marker='_', lw=18, label='unhealthy for \n sensitive group')
    ax[2].plot(np.arange(85, 105), np.ones(len(np.arange(85, 105))),
        c='red', marker='_', lw=18, label='unhealthy')
    ax[2].plot(np.arange(105, 200), np.ones(len(np.arange(105, 200))),
        c='purple', marker='_', lw=18, label='very unhealthy')

    ax[2].plot(np.arange(0, 35), np.zeros(len(np.arange(0, 35))),
        c='royalblue', marker='_', lw=18, label='very good')
    ax[2].plot(np.arange(35, 50), np.zeros(
    len(np.arange(35, 50))), c='green', marker='_', lw=18)
    ax[2].plot(np.arange(50, 70), np.zeros(len(np.arange(50, 70))),
        c='yellow', marker='_', lw=18)
    ax[2].plot(np.arange(70, 120), np.zeros(len(np.arange(70, 120))),
        c='orange', marker='_', lw=18)
    ax[2].plot(np.arange(120, 200), np.zeros(
    len(np.arange(120, 200))), c='red', marker='_', lw=18)
    ax[2].set_xlabel('ppb')

    #--AQI label--(TH)
    ax[2].text(22,-0.2,'25')
    ax[2].text(38,-0.2,'50') 
    ax[2].text(55,-0.2,'100')
    ax[2].text(105,-0.2,'200')
    #ax[0].text(400,-0.2,'323')

    #--AQI label--(US)
    ax[2].text(42,0.8,'50') 
    ax[2].text(55,0.8,'100')
    ax[2].text(70,0.8,'150')
    ax[2].text(90,0.8,'200')
    #ax[1].text(405,0.8,'300')

    #-----------------------CO---------------------------

    ax[3].set_title('CO(8hr avg) AQI converstion')
    ax[3].plot(np.arange(0, 4.4), np.ones(len(np.arange(0, 4.4))),
        c='green', marker='_', lw=18, label='good/satisfactory')
    ax[3].plot(np.arange(4.4, 9.4), np.ones(len(np.arange(4.4,  9.4))),
        c='yellow', marker='_', lw=18, label='moderate')
    ax[3].plot(np.arange(9.4, 12.5), np.ones(len(np.arange(9.4, 12.5))),
        c='orange', marker='_', lw=18, label='unhealthy for \n sensitive group')
    ax[3].plot(np.arange(12.4, 15.4), np.ones(len(np.arange(12.4, 15.4))),
        c='red', marker='_', lw=18, label='unhealthy')
    ax[3].plot(np.arange(15.4, 30.4), np.ones(len(np.arange(15.4, 30.4))),
        c='purple', marker='_', lw=18, label='very unhealthy')

    ax[3].plot(np.arange(0, 4.4), np.zeros(len(np.arange(0, 4.4))),
        c='royalblue', marker='_', lw=18, label='very good')
    ax[3].plot(np.arange(4.4, 6.4), np.zeros(
    len(np.arange(4.4, 6.4))), c='green', marker='_', lw=18)
    ax[3].plot(np.arange(6.4, 9.0), np.zeros(len(np.arange(6.4, 9.0))),
        c='yellow', marker='_', lw=18)
    ax[3].plot(np.arange(9.0, 30), np.zeros(len(np.arange(9.0, 30))),
        c='orange', marker='_', lw=18)
    ax[3].plot(np.arange(30, 40), np.zeros(
    len(np.arange(30, 40))), c='red', marker='_', lw=18)
    ax[3].set_xlabel('ppm')

    #--AQI label--(TH)
    ax[3].text(2,-0.2,'25')
    ax[3].text(4.5,-0.2,'50') 
    ax[3].text(6.4,-0.2,'100')
    ax[3].text(27,-0.2,'200')
    #ax[0].text(400,-0.2,'323')

    #--AQI label--(US)
    ax[3].text(2,0.8,'50') 
    ax[3].text(6.5,0.8,'100')
    ax[3].text(9.5,0.8,'150')
    ax[3].text(12.5,0.8,'200')

    #-----------------------NO2---------------------------
    ax[4].set_title('NO$_2$(1hr avg) AQI converstion')

    ax[4].plot(np.arange(0, 53), np.ones(len(np.arange(0, 53))),
        c='green', marker='_', lw=18, label='good/satisfactory')
    ax[4].plot(np.arange(53, 100), np.ones(len(np.arange(53,  100))),
        c='yellow', marker='_', lw=18, label='moderate')
    ax[4].plot(np.arange(100, 360), np.ones(len(np.arange(100, 360))),
        c='orange', marker='_', lw=18, label='unhealthy for \n sensitive group')
    ax[4].plot(np.arange(360, 649), np.ones(len(np.arange(360, 649))),
        c='red', marker='_', lw=18, label='unhealthy')
    ax[4].plot(np.arange(649, 1244), np.ones(len(np.arange(649, 1244))),
        c='purple', marker='_', lw=18, label='very unhealthy')


    # ---- Thailand ----------------
    ax[4].plot(np.arange(0, 60), np.zeros(len(np.arange(0, 60))),
        c='royalblue', marker='_', lw=18, label='very good')
    ax[4].plot(np.arange(60, 106), np.zeros(
    len(np.arange(60, 106))), c='green', marker='_', lw=18)
    ax[4].plot(np.arange(106, 170), np.zeros(len(np.arange(106, 170))),
        c='yellow', marker='_', lw=18)
    ax[4].plot(np.arange(170, 340), np.zeros(len(np.arange(170, 340))),
        c='orange', marker='_', lw=18)
    ax[4].plot(np.arange(340, 1244), np.zeros(
    len(np.arange(340, 1244))), c='red', marker='_', lw=18)
    ax[4].set_xlabel('ppb')

    #--AQI label--(TH)
    ax[4].text(-1,-0.2,'25',ha='left')
    ax[4].text(44,-0.2,'50',ha='left') 
    ax[4].text(90,-0.2,'100')
    ax[4].text(242,-0.2,'200')
     

    #--AQI label--(US)
    ax[4].text(-1,0.8,'50',ha='left') 
    ax[4].text(30,0.8,'100',ha='left')
    ax[4].text(255,0.8,'150')
    ax[4].text(552,0.8,'200')

    #  -----------------------SO2---------------------------

    ax[5].set_title('SO$_2$(1hr avg) AQI converstion')

    ax[5].plot(np.arange(0, 35), np.ones(len(np.arange(0, 35))),
        c='green', marker='_', lw=18, label='good/satisfactory')
    ax[5].plot(np.arange(35, 75), np.ones(len(np.arange(35,  75))),
        c='yellow', marker='_', lw=18, label='moderate')
    ax[5].plot(np.arange(75, 185), np.ones(len(np.arange(75, 185))),
        c='orange', marker='_', lw=18, label='unhealthy for \n sensitive group')
    ax[5].plot(np.arange(185, 304), np.ones(len(np.arange(185, 304))),
        c='red', marker='_', lw=18, label='unhealthy')
    ax[5].plot(np.arange(304, 604), np.ones(len(np.arange(304, 604))),
        c='purple', marker='_', lw=18, label='very unhealthy')

    ax[5].plot(np.arange(0, 100), np.zeros(len(np.arange(0, 100))),
        c='royalblue', marker='_', lw=18, label='very good')
    ax[5].plot(np.arange(100, 200), np.zeros(
        len(np.arange(100, 200))), c='green', marker='_', lw=18)
    ax[5].plot(np.arange(200, 300), np.zeros(len(np.arange(200, 300))),
        c='yellow', marker='_', lw=18)
    ax[5].plot(np.arange(300, 400), np.zeros(len(np.arange(300, 400))),
        c='orange', marker='_', lw=18)
    ax[5].plot(np.arange(400, 604), np.zeros(
        len(np.arange(400, 604))), c='red', marker='_', lw=18)
    ax[5].set_xlabel('ppb')


    #--AQI label--(TH)
    ax[5].text(65,-0.2,'25')
    ax[5].text(165,-0.2,'50') 
    ax[5].text(255,-0.2,'100')
    ax[5].text(352,-0.2,'200')
    #ax[0].text(400,-0.2,'323')

    #--AQI label--(US)
    ax[5].text(5,0.8,'50') 
    ax[5].text(29,0.8,'100')
    ax[5].text(142,0.8,'150')
    ax[5].text(257,0.8,'200')

    ax[0].legend(bbox_to_anchor=(1.05, 1.05),frameon=False)


    labels = [item.get_text() for item in ax[0].get_yticklabels()]
    labels[1] = 'Thai AQI'
    labels[2] = 'US AQI'
    #labels[3] = 'My Level'

    for a in ax:
        a.set_ylim([-1, 2])
        a.set_yticklabels(labels)


    plt.tight_layout()
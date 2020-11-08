# -*- coding: utf-8 -*-
from ..imports import *
from ..gen_functions import *


def plot_dendogram(
        data: pd.core.frame.DataFrame,
        cols=None,
        front_size=16,
        filename=None):
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
    length = len(index_list)
    if length == 3:
        colors = ['royalblue', 'orange', 'red']
        label_list = ['Training', 'Validation', 'Test']

    elif length == 4:
        colors = ['royalblue', 'orange', 'red', 'maroon']
        label_list = ['Training', 'Validation1', 'Validation2', 'Test']

    elif length == 2:
        colors = ['royalblue',  'red' ]
        label_list = ['Training', 'Test']

    else:
        colors = get_color(color_length=length, cmap=cm.jet)
        label_list = []

    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax_list = []
    for idxs, color in zip(index_list, colors):
        # Plot training and test indices
        l1 = ax.scatter(idxs, [1] * len(idxs), c=color, marker='_', lw=6)
        ax_list.append(l1)

    ax.legend(ax_list, label_list)
    ax.set(ylim=[0.8, 1.5], title='Split behavior', xlabel='date')
    plt.xticks(rotation=90)


def plot_corr(df, cor_method='pearson', figsize=(8, 6), filename=None):
    """Plot the correlation between different pollutants

    Args:
        poll_df:
        core_method
        filename

    """
    #df = poll_df.resample(avg).max()
    plt.figure(figsize=figsize)
    mask = np.tril(df.corr())
    # use spearman rank correlation
    sns.heatmap(df.corr(method=cor_method), annot=True, mask=mask)

    if filename:
        plt.savefig(filename)


def plot_season_avg(
        poll_df,
        pollutant,
        ax,
        plot_error=True,
        roll=True,
        agg='max',
        color='blue',
        linestyle='solid',
        linewidth=2, label=None):
    """Plot the average by date of year. Good for looking seasonal pattern.

    Args:
        poll_df: dataframe for plotting the data. Must have datetime index
        pollutant: columns to plot
        ax: axis object to plot
        plot_error: if True, use sns.lineplot to show the error
        roll: if True, calculate the rolling average or use daily average
        agg: either 'max' or 'mean'
        label(optional): label word 

    """
    plt.rcParams.update({'font.size': 14})

    # if roll:
    #     df = poll_df[[pollutant]].rolling(24, min_periods=None).agg('mean').copy().dropna()

    # df = poll_df[[pollutant]].resample('d').agg(agg).copy().dropna()

    # df['dayofyear'] = df.index.dayofyear
    # df['year'] = df.index.year

    # # add winter day by substratcing the first day of july
    # winterday = df['dayofyear'] - 182
    # # get rid of the negative number
    # winter_day_max = winterday.max()
    # winterday[winterday < 0] = winterday[winterday < 0] + 182 + winter_day_max
    # df['winter_day'] = winterday

    # # add month-day
    # df['month_day'] = df.index.strftime('%m-%d')
    # temp = df[['winter_day', 'month_day']].set_index('winter_day')
    # temp.index = temp.index.astype(str)
    # winter_day_dict = temp.to_dict()['month_day']

    df, winter_day_dict = season_avg(
        poll_df, cols=[pollutant], roll=roll, agg=agg, offset=182)
    
    if label==None:
        label=pollutant

    if plot_error:
        sns.lineplot(
            data=df,
            x='winter_day',
            y=pollutant,
            ax=ax,
            legend='brief',
            label=label,
            color=color)

    else:
        mean_day = df.groupby('winter_day').mean()[pollutant]
        ax.plot(
            mean_day,
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle)

    ax.set_xlim([0, 366])
    new_ticks = [
        '07-01',
        '08-20',
        '10-09',
        '11-28',
        '01-16',
        '03-06',
        '04-25',
        '06-14',
        '']

    ax.set_xticklabels(new_ticks)
    ax.legend()
    ax.set_xlabel('month-date')
    # plt.show()
    return winter_day_dict, df.groupby('winter_day').mean()[pollutant]


def plot_all_pollutions(
        poll_df,
        city_name='',
        filename=None,
        transition_dict=None,
        color_labels=None,
        level_name=None):
    """Plot all pollutant data over time.

    Args:
        poll_df
        city_name
        filename=None
        transition_dict=None
        color_labels=None
        level_name=None

    """
    if transition_dict is None:
        transition_dict = {'PM2.5': [0, 12, 35.5, 55.4, 150.4, 1e3],
                           'PM10': [0, 154, 254, 354, 504, 1e3],
                           'O3': [0, 54, 70, 85, 105, 200],
                           'SO2': [0, 35, 75, 185, 304, 1e3],
                           'NO2': [0, 53, 100, 360, 649, 1e3],
                           'CO': [0, 4.4, 9.4, 12.5, 15.4, 1e3]}

    if color_labels is None:
        color_labels = ['green', 'goldenrod','orange', 'red', 'purple']

    if level_name is None:
        level_names = [
            'good',
            'moderate', 
            'unhealthy(sen)',
            'unhealthy',
            'very unhealthy']

    gas_list = poll_df.columns
    print('pollutants to plot', gas_list)
    len_gas = len(gas_list)

    _, ax = plt.subplots(len_gas, 1, figsize=(10, 3 * len_gas), sharex=True)

    if len_gas > 1:
        # obtain daily avg to plot
        d_avg = poll_df.resample('d').mean()

        for i, a in enumerate(ax):
            col = gas_list[i]
            poll_data = poll_df[[col]].dropna()
            poll_data['color'] = pd.cut(
                poll_data[col],
                bins=transition_dict[col],
                labels=color_labels)

            for color, legend in zip(color_labels, level_names):
                temp = poll_data[poll_data['color'] == color]
                # plot the data for each pollution level
                a.scatter(
                    temp.index,
                    temp[col],
                    c=temp['color'],
                    marker='.',
                    label=legend,
                    alpha=0.7)

            if col in ['PM2.5']:
                a.set_ylabel(col + r'($\mu g/m^3$)')
                a.axhline(transition_dict[col][3], color='red', linestyle='dashed')
                a.axhline(transition_dict[col][4], color='purple', linestyle='dashed' )

            elif col in ['PM10']:
                a.set_ylabel(col + r'($\mu g/m^3$)')
                a.axhline(transition_dict[col][3], color='red', linestyle='dashed')
                a.axhline(transition_dict[col][4], color='purple', linestyle='dashed' )
            elif col in ['O3']:
                a.set_ylabel(col + '(ppb)')
                a.axhline(transition_dict[col][3], color='red', linestyle='dashed')
                a.axhline(transition_dict[col][4], color='purple', linestyle='dashed' )

            elif col in ['NO2', 'SO2']:
                a.set_ylabel(col + '(ppb)')
            elif col == 'CO':
                a.set_ylabel(col + '(ppm)')

            a.axhline(transition_dict[col][2], color='orange', linestyle='dashed')
            a.axhline(transition_dict[col][1], color='goldenrod', linestyle='dashed')

            a.plot(d_avg[col], label='avg' + col, color='black', alpha=0.7)
            a.legend(loc='upper left')
            # if i in [0, 1, 2]:
            #    a.axhline(transition_dict[col][2], color='red')
            # if i in [0, 1, 2]:
            #    a.axhline(transition_dict[col][3], color='purple')

        #ax[0].legend(loc='upper left')
        ax[0].set_xlim([poll_df.index.min(), poll_df.index.max()])
        ax[0].set_title(
            f'Pollutants Data for {city_name} Averaged from All Staions')
        ax[-1].set_xlabel('date')

    else:
        col = gas_list[0]
        a = ax

        poll_data = poll_df[[col]].dropna()
        poll_data['color'] = pd.cut(poll_data[col],
                                    bins=transition_dict[col],
                                    labels=color_labels)

        for color, legend in zip(color_labels, level_names):
            temp = poll_data[poll_data['color'] == color]
            # plot the data for each pollution level
            a.scatter(
                temp.index,
                temp[col],
                c=temp['color'],
                marker='.',
                label=legend,
                alpha=0.7)

        temp = poll_df.resample('d').mean()
        a.plot(
            temp.index,
            temp[col],
            label='avg' + col,
            color='black',
            alpha=0.7)
        a.legend(loc='upper left')

        if col in ['PM2.5']:
            a.set_ylabel(col + r'($\mu g/m^3$)')
            a.axhline(transition_dict[col][3], color='red', linestyle='dashed')
            a.axhline(transition_dict[col][4], color='purple', linestyle='dashed' )
            #a.axhline(transition_dict[col][5], color='purple', linestyle='dashed')
        elif col in ['PM10']:
            a.set_ylabel(col + r'($\mu g/m^3$)')
            a.axhline(transition_dict[col][3], color='red', linestyle='dashed')
            a.axhline(transition_dict[col][4], color='purple', linestyle='dashed')
        elif col in ['O3']:
            a.set_ylabel(col + '(ppb)')
            a.axhline(transition_dict[col][3], color='red', linestyle='dashed')
            a.axhline(transition_dict[col][4], color='purple', linestyle='dashed')

        elif col in ['NO2', 'SO2']:
            a.set_ylabel(col + '(ppb)')
        elif col == 'CO':
            a.set_ylabel(col + '(ppm)')

        a.axhline(transition_dict[col][2], color='orange', linestyle='dashed')
        a.axhline(transition_dict[col][1], color='goldenrod', linestyle='dashed')

    plt.tight_layout()

    if filename:
        plt.savefig(filename)


def poll_to_aqi(poll_df, roll_dict):
    """Convert concentration pollution dataframe to aqi dataframe

    Args:
        poll_df : pollution data
        rolling_dict: rolling information for each pollutant

    Returns: aqi dataframe

    """
    for pollutant in poll_df.columns:
        if pollutant in roll_dict.keys():
            rolling_win = roll_dict[pollutant]
            poll_df[pollutant] = poll_df[pollutant].rolling(
                rolling_win, min_periods=0).mean().round(1)
            # convert to aqi
            poll_df[pollutant] = poll_df[pollutant].apply(
                to_aqi, pollutant=pollutant)

    return poll_df


def plot_polls_aqi(
        poll_df,
        roll_dict,
        city_name='',
        filename=None,
        color_labels=None,
        level_name=None):
    """Plot all pollutant data over time.

    Args:
        poll_df
        rolling_dict
        city_name
        filename=None
        transition_dict=None
        color_labels=None
        level_name=None

    """
    # convert to aqi
    poll_df = poll_to_aqi(poll_df, roll_dict)

    # convert to daily average
    poll_df = poll_df.resample('d').mean()
    # reorder the data according to average AQI
    new_cols = poll_df.mean(
        axis=0).sort_values(
        ascending=False).index.to_list()
    poll_df = poll_df[new_cols]
    if len(new_cols) > 1:
        length = int(len(new_cols) / 2)

    levels = [50, 100, 150, 200, 300]
    text_pos = [25, 75, 125, 175, 250]
    color_labels = [ 'goldenrod', 'orange', 'red', 'purple', 'purple']
    level_names = [
        ' good',
        ' moderate', ' unhealthy(sen)',
        ' unhealthy',
        ' very unhealthy']
    #data_colors = get_color(color_length=len(new_cols), cmap=cm.brg)
    data_colors = get_gas_color_list(new_cols)

    if len(new_cols) > 1:
        # more than one data, plot in two subplots
        _, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for i, a in enumerate(ax):
            temp = poll_df[new_cols[length * i:length * (i + 1)]]
            colors = data_colors[length * i:length * (i + 1)]
            for col, color in zip(
                    new_cols[length * i:length * (i + 1)], colors):
                a.plot(
                    temp[col],
                    marker='.',
                    markersize=1,
                    linewidth=2,
                    alpha=0.6,
                    color=color)
            a.legend(temp.columns, loc='upper left')
            if i == 0:
                a.set_title(
                    f'Daily Average AQI in {city_name} for different pollutants')
            else:
                a.set_xlabel('date')

            # select ymax to select level
            ymax = temp.max().max()
            try:
                idx = np.where(levels < ymax)[0][-1]
            except:
                idx = 0
            idx += 1
            # make horizontal line
            for l, c, n, p in zip(
                    levels[:idx], color_labels[:idx], level_names[:idx], text_pos):
                a.axhline(l, color=c, label=n, linestyle='dashed')
                a.text(temp.index.max(), p, n, horizontalalignment='left')

            a.set_xlim([temp.index.min(), temp.index.max()])
            a.set_ylim([0, ymax])
            a.set_ylabel('AQI')
    else:
        # only one data only one subplots
        # more than one data, plot in two subplots
        _, a = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        a.plot(poll_df, color=data_colors[0])
        a.legend(poll_df.columns, loc='upper left')
        a.set_title(
            f'Daily Average AQI in {city_name} for different pollutants')
        a.set_xlabel('date')
        # make horizontal line
        ymax = poll_df.max().values[0]
        idx = np.where(levels < ymax)[0][-1] + 1
        for l, c, n in zip(
                levels[:idx], color_labels[:idx], level_names[:idx]):
            a.axhline(l, color=c, label=n)
            a.text(poll_df.index.max(), l, n, horizontalalignment='left')
        a.set_xlim([poll_df.index.min(), poll_df.index.max()])
        a.set_ylim([0, ymax])
        a.set_ylabel('AQI')
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    plt.tight_layout()

    if filename:
        plt.savefig(filename)


def plot_season_aqi(
        poll_df,
        roll_dict,
        pollutant,
        filename=None,
        aqi_line=True, aqi_text=True):
    """Plot average seasonal AQI value of a pollutant, and identify the high months.

    Args:
        poll_df: raw pollution data
        roll_dict: rolling dictionary
        pollutant: pollutant value
        filename(optional): filename to save
        aqi_line(optional): if True, show horizontal aqi line
        aqi_text(optional): if True, show aqi text

    """
    _, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)

    if aqi_line:
        # aqiline
        ax.axhline(50, color='goldenrod', linestyle='dashed')
        ax.axhline(100, color='orange', linestyle='dashed')
        ax.axhline(150, color='red', linestyle='dashed')
        ax.axhline(200, color='purple', linestyle='dashed')
        #ax.text(365, 100, ' moderate', horizontalalignment='left')
        #ax.text(365, 150, ' unhealthy', horizontalalignment='left')
    if aqi_text:
        ax.text(365, 40, ' good', horizontalalignment='left')
        ax.text(365, 75, ' moderate', horizontalalignment='left')
        ax.text(365, 125, ' unhealthy(sen)', horizontalalignment='left')
        ax.text(365, 175, ' unhealthy', horizontalalignment='left')

    poll_aqi = poll_to_aqi(poll_df, roll_dict)
    winter_day_dict, mean_day = plot_season_avg(poll_aqi, pollutant, ax, plot_error=True, roll=False)

    ax.set_ylabel('AQI')

    temp = mean_day[mean_day > 100]
    if len(temp) > 0:
        print('aqi 100 in ',
              winter_day_dict[str(temp.index.min())],
              'to',
              winter_day_dict[str(temp.index.max())])

    temp = mean_day[mean_day > 150]
    if len(temp) > 0:
        print('aqi 150 in ',
              winter_day_dict[str(temp.index.min())],
              'to',
              winter_day_dict[str(temp.index.max())])

    plt.tight_layout()

    if filename:
        plt.savefig(filename)

    return ax, winter_day_dict


def cal_sea_yr(df, agg='mean', start_month='-12-01', end_month='-04-30'):
    """Calculate the season year average.

    Args:
        df: data to calculate the season average. Must have datetime index
        agg: aggeration method (ex 'mean')
        start_month: starting month of the season
        end_month: ending month of the season

    Returns: pd.DataFrame
        season yearly average

    """
    df = add_season(df, start_month=start_month, end_month=end_month)
    # remove other season
    df = df[df['season'] != 'other'].drop('season', axis=1)
    return df.groupby('year').agg(agg)


def add_ln_trend_line(
        series,
        ax,
        color='royalblue',
        linewidth=1,
        linestyle='dashed'):
    """Fit linear line between the index of df and the data and add linear trend line to the plot.

    Args:
        series: series to lienar fit
        ax: plt axis object
        color: color of the line
        linewidth: width of the line
        linestyle: linestyle

    Returns: np.array, np.array
        numpy array of the linear trend line x and y

    """
    # linear fit the data
    x = series.index.values
    y = series.values

    slope, intercept, *args = linregress(x, y)
    z = np.array([slope, intercept])
    #z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    # string for labeling
    z_str = z.round(2).astype(str)
    z_str[1] = z[1].astype(int).astype(str)
    if z[1] > 0:
        z_str[1] = '+' + z_str[1]
    # create a trend line
    y = p(x)
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=f'({z_str[0]})x{z_str[1]}')

    return x, y


def plot_chem_print(
        sum_df,
        city_name,
        color_list=None,
        ylim=[
            0,
            90],
        filename=None):
    """Bar plot the average (or max) AQI of each gas to show the AQI Fingerprint of that city.

    Args:
        sum_df: a series of satistical summary. For example poll_df.mean(axis=0)
        city_name: city name for title
        color_list:a list of color to use. If None, use the
        ylim: yaxis range
        filename(optional): filename to save data

    """
    gas_list = ['PM2.5', 'PM10', 'O3', 'CO', 'NO2', 'SO2']
    gas_list = [g for g in gas_list if g in sum_df.index]
    sum_df = sum_df.loc[gas_list]

    if color_list is None:
        color_list = get_gas_color_list(gas_list)

    plt.figure(figsize=(7, 3))
    plt.bar(
        sum_df.index,
        sum_df,
        edgecolor='black',
        color=color_list,
        width=0.6)
    plt.ylim(ylim)
    plt.title(f'AQI Fingerprint for {city_name}')
    plt.xlabel('pollutant')
    plt.ylabel('AQI')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)


def plot_yearly_ln(dataset, min_year=None, filename=None, start_month='-10-01', end_month='-04-30' ):
    """Obtain yearly trends of PM pollutant data, number of hotspots and temperatures to compare their trends.

    Args:
        dataset: dataset object with all data
        filename(optional): filename to save data
        start_month: starting month of the season
        end_month: ending month of the season

    """
    year_fire = cal_sea_yr(
        dataset.fire.resample('d').sum()[
            ['count']].copy(), agg='mean')
    year_fire.columns = ['number of hotspots']

    year_temp = cal_sea_yr(
        dataset.wea[['Temperature(C)']].resample('d').mean().copy())

    if 'PM10' in dataset.poll_df.columns:
        poll_col = ['PM2.5', 'PM10']
        y_labels = [r'$\mu g/m^3$', r'$\mu g/m^3$', 'counts/day', '($^o$C)']
        colors = ['royalblue', 'green', 'red', 'orange']
    else:
        poll_col = ['PM2.5']
        y_labels = [r'$\mu g/m^3$', 'counts/day', '($^o$C)']
        colors = ['royalblue', 'red', 'orange']

    year_poll = cal_sea_yr(
        dataset.poll_df.resample('d').mean().copy(), start_month=start_month, end_month=end_month)[poll_col]
    if min_year is None:
        min_year = year_fire.index.min()

    year_avg = pd.concat(
        [year_poll.loc[min_year:], year_fire.loc[min_year:], year_temp.loc[min_year:]], axis=1)

    _, ax = plt.subplots(
        len(y_labels), 1, figsize=(
            10, 3 * len(y_labels)), sharex=True)

    for i, (a, col, y_label, color) in enumerate(
            zip(ax, year_avg.columns, y_labels, colors)):
        a.plot(year_avg[col], marker='*', label=col, color=color)
        a.set_ylabel(y_label)

        try:
            # add linear trend line an display equation
            x, _ = add_ln_trend_line(year_avg[col].dropna(), a, color=color)
        except BaseException:
            pass
        a.legend(loc='upper left')

        if i == 0:
            a.set_title(
                'Trend of Pollutions, Fire Activities, and Tempearatures')

    a.set_xlabel('year(pollution season)')
    a.xaxis.set_major_locator(MaxNLocator(integer=True))

    if filename:
        plt.savefig(filename)

    return ax, year_avg


def compare_seson_avg(
        dataset,
        poll='PM2.5',
        wea_col=[
            'Temperature(C)',
            'Wind_Speed(kmph'],
    agg='mean',
        filename=None):
    """Compare seasonal pattern of pollution data, fire and weather pattern.

    Args:
        dataset: dataset object
        poll(optional): pollutant name
        wea_col(optional): a list of weather columns For example
        agg(optional): aggegration method for pollution
        filename(optional): filename to save data

    """
    plot_length = len(wea_col) + 2

    _, ax = plt.subplots(
        plot_length, 1, figsize=(
            10, 3 * plot_length), sharex=True)

    winter_day_dict, mean_day = plot_season_avg(dataset.poll_df.copy(
    ), poll, ax[0], plot_error=True, roll=True, agg=agg, linewidth=2)
    ax[0].set_ylabel(get_unit(poll))
    # aqiline
    ax[0].axhline(12, color='goldenrod', linestyle='dashed')
    ax[0].axhline(35.4, color='orange', linestyle='dashed')
    #ax[0].axhline(55.4, color='red', linestyle='dashed')
    ax[0].axhline(55.4, color='red', linestyle='dashed')
    ax[0].axhline(150.4, color='purple', linestyle='dashed')
    #ax[0].text(365, 35.4, ' moderate', horizontalalignment='left')
    #ax[0].text(365, 55.4, ' unhealthy', horizontalalignment='left')
    ax[0].text(365, 6, ' good', horizontalalignment='left')
    ax[0].text(365, 20, ' moderate', horizontalalignment='left')
    ax[0].text(365, 43, ' unhealthy(sen)', horizontalalignment='left')
    ax[0].text(365, 100.4, ' unhealthy', horizontalalignment='left')

    fire_hour = dataset.fire[['count']].resample('d').sum()
    fire_hour.columns = ['number of hotspots']
    winter_day_dict, fire_mean_day = plot_season_avg(fire_hour.copy(
    ), 'number of hotspots', ax[1], plot_error=True, roll=False, agg='mean', color='red', linestyle='solid', linewidth=2)
    ax[1].set_ylabel('number of hotspot')

    t_hour = dataset.wea[wea_col].resample('d').mean().copy()

    for i, col in enumerate(wea_col):

        winter_day_dict, temperature = plot_season_avg(t_hour.copy(
        ), col, ax[i + 2], plot_error=True, roll=False, agg='mean', color='orange', linestyle='solid', linewidth=2)
        ax[i + 2].set_ylabel('($^o$C)')

    for a in ax:
        a.legend(loc='upper left')

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
    
    return ax


def plot_hour_avg(df, col, ax, color='blue', label=None):
    """Plot average hourly behavior. Must have datetime index.

    Args:
        df: dataframe containing the column to plot
        col: name of the columns to plot
        ax: axis to plot the data on
        color(optional): color of the plot

    """
    df = df.resample('h').mean()
    # add hour of day from the index.
    df['time_of_day'] = df.index.hour

    if label==None:
        label = col

    sns.lineplot(
        data=df,
        x='time_of_day',
        y=col,
        color=color,
        ax=ax,
        legend='brief',
        label=label)
    ax.set_ylabel(col)
    ax.set_xlabel('hour')
    #ax[2].set_title('hourly fire activities')
    
    ax.legend(loc='upper left')


def compare_us_thai_aqi():
    """Plot the different between US and Thailand aqi conversion.

    """
    plt.rcParams.update({'font.size': 14})
    _, ax = plt.subplots(6, 1, figsize=(12, 15))

    ax[0].set_title('PM2.5(24hr avg) AQI converstion')
    # Thailand
    ax[0].plot(np.arange(0, 25), np.zeros(len(np.arange(0, 25))),
               c='royalblue', marker='_', lw=18, label='very good')

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
    ax[0].set_xlabel(r'PM2.5($\mu$g/m$^3$)')

    # --AQI label--(TH)
    ax[0].text(13, -0.2, '25')
    ax[0].text(26, -0.2, '50')
    ax[0].text(35, -0.2, '100')
    ax[0].text(75, -0.2, '200')
    # ax[0].text(190,-0.2,'310')

    # --AQI label--(US)
    ax[0].text(0, 0.8, '50')
    ax[0].text(22, 0.8, '100')
    ax[0].text(40, 0.8, '150')
    ax[0].text(165, 0.8, '200')
    # ax[0].text(192,0.8,'300')

    # -----------------------PM10---------------------------

    ax[1].set_title('PM10(24hr avg) AQI converstion')

    # -------------us-----------------
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

    # -------------Thailand -------------------
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
    ax[1].set_xlabel(r'PM10($\mu$g/m$^3$)')

    # --AQI label--(TH)
    ax[1].text(25, -0.2, '25')
    ax[1].text(58, -0.2, '50')
    ax[1].text(90, -0.2, '100')
    ax[1].text(150, -0.2, '200')
    # ax[0].text(400,-0.2,'323')

    # --AQI label--(US)
    ax[1].text(25, 0.8, '50')
    ax[1].text(125, 0.8, '100')
    ax[1].text(225, 0.8, '150')
    ax[1].text(325, 0.8, '200')
    # ax[1].text(405,0.8,'300')

    # -----------------------O3---------------------------

    ax[2].set_title('O$_3$(8hr avg) AQI converstion')

    ax[2].plot(np.arange(0, 54), np.ones(len(np.arange(0, 54))),
               c='green', marker='_', lw=18, label='good/satisfactory')
    ax[2].plot(np.arange(54, 70), np.ones(len(np.arange(54, 70))),
               c='yellow', marker='_', lw=18, label='moderate')
    ax[2].plot(np.arange(70, 85), np.ones(len(np.arange(70, 85))), c='orange',
               marker='_', lw=18, label='unhealthy for \n sensitive group')
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

    # --AQI label--(TH)
    ax[2].text(22, -0.2, '25')
    ax[2].text(38, -0.2, '50')
    ax[2].text(55, -0.2, '100')
    ax[2].text(105, -0.2, '200')
    # ax[0].text(400,-0.2,'323')

    # --AQI label--(US)
    ax[2].text(42, 0.8, '50')
    ax[2].text(55, 0.8, '100')
    ax[2].text(70, 0.8, '150')
    ax[2].text(90, 0.8, '200')
    # ax[1].text(405,0.8,'300')

    # -----------------------CO---------------------------

    ax[3].set_title('CO(8hr avg) AQI converstion')
    ax[3].plot(np.arange(0, 4.4), np.ones(len(np.arange(0, 4.4))),
               c='green', marker='_', lw=18, label='good/satisfactory')
    ax[3].plot(np.arange(4.4, 9.4), np.ones(len(np.arange(4.4, 9.4))),
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

    # --AQI label--(TH)
    ax[3].text(2, -0.2, '25')
    ax[3].text(4.5, -0.2, '50')
    ax[3].text(6.4, -0.2, '100')
    ax[3].text(27, -0.2, '200')
    # ax[0].text(400,-0.2,'323')

    # --AQI label--(US)
    ax[3].text(2, 0.8, '50')
    ax[3].text(6.5, 0.8, '100')
    ax[3].text(9.5, 0.8, '150')
    ax[3].text(12.5, 0.8, '200')

    # -----------------------NO2---------------------------
    ax[4].set_title('NO$_2$(1hr avg) AQI converstion')

    ax[4].plot(np.arange(0, 53), np.ones(len(np.arange(0, 53))),
               c='green', marker='_', lw=18, label='good/satisfactory')
    ax[4].plot(np.arange(53, 100), np.ones(len(np.arange(53, 100))),
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

    # --AQI label--(TH)
    ax[4].text(-1, -0.2, '25', ha='left')
    ax[4].text(44, -0.2, '50', ha='left')
    ax[4].text(90, -0.2, '100')
    ax[4].text(242, -0.2, '200')

    # --AQI label--(US)
    ax[4].text(-1, 0.8, '50', ha='left')
    ax[4].text(30, 0.8, '100', ha='left')
    ax[4].text(255, 0.8, '150')
    ax[4].text(552, 0.8, '200')

    #  -----------------------SO2---------------------------

    ax[5].set_title('SO$_2$(1hr avg) AQI converstion')

    ax[5].plot(np.arange(0, 35), np.ones(len(np.arange(0, 35))),
               c='green', marker='_', lw=18, label='good/satisfactory')
    ax[5].plot(np.arange(35, 75), np.ones(len(np.arange(35, 75))),
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

    # --AQI label--(TH)
    ax[5].text(65, -0.2, '25')
    ax[5].text(165, -0.2, '50')
    ax[5].text(255, -0.2, '100')
    ax[5].text(352, -0.2, '200')
    # ax[0].text(400,-0.2,'323')

    # --AQI label--(US)
    ax[5].text(5, 0.8, '50')
    ax[5].text(29, 0.8, '100')
    ax[5].text(142, 0.8, '150')
    ax[5].text(257, 0.8, '200')

    ax[0].legend(bbox_to_anchor=(1.05, 1.05), frameon=False)

    labels = [item.get_text() for item in ax[0].get_yticklabels()]
    labels[1] = 'Thai AQI'
    labels[2] = 'US AQI'
    

    for a in ax:
        a.set_ylim([-1, 2])
        a.set_yticklabels(labels)

    plt.tight_layout()



def compare_aqis(filename=None):
    """Plot the different between US and Thailand aqi conversion and convention in this project.

    """
    plt.rcParams.update({'font.size': 14})
    _, ax = plt.subplots(2, 1, figsize=(12, 6))

    ax[0].set_title('PM2.5(24hr avg) AQI converstion')
    # Thailand
    ax[0].plot(np.arange(0, 25), np.zeros(len(np.arange(0, 25))),
               c='royalblue', marker='_', lw=18, label='very good')

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
    ax[0].set_xlabel(r'$\mu$g/m$^3$')

    ## simplified system
    #ax[0].plot(np.arange(0, 12), np.ones(len(np.arange(0, 12)))+1,
    #           c='green', marker='_', lw=18 )
    #ax[0].plot(np.arange(12, 35.5), np.ones(len(np.arange(12, 35.5)))+1,
    #           c='orange', marker='_', lw=18 )
    #ax[0].plot(np.arange(35.5, 55.5), np.ones(len(np.arange(35.5, 55.5)))+1,
    #           c='red', marker='_', lw=18)
    #ax[0].plot(np.arange(55.5, 180.5), np.ones(len(np.arange(55.5, 180.5)))+1,
    #           c='red', marker='_', lw=18 )
    #ax[0].plot(np.arange(180.5, 200.0), np.ones(len(np.arange(180.5, 200)))+1,
    #           c='purple', marker='_', lw=18)

    # --AQI label--(TH)
    ax[0].text(13, 0, '25', verticalalignment='center')
    ax[0].text(26, 0, '50', verticalalignment='center')
    ax[0].text(35, 0, '100', verticalalignment='center')
    ax[0].text(75, 0, '200', verticalalignment='center')
    # ax[0].text(190,0, , verticalalignment='center''310')

    # --AQI label--(US)
    ax[0].text(0, 1, '50', verticalalignment='center')
    ax[0].text(22, 1, '100', verticalalignment='center')
    ax[0].text(40, 1, '150', verticalalignment='center')
    ax[0].text(165, 1, '200', verticalalignment='center')
    # ax[0].text(192,0.8,'300')

    #  -----------------------SO2---------------------------

    ax[1].set_title('SO$_2$(1hr avg) AQI converstion')

    ax[1].plot(np.arange(0, 35), np.ones(len(np.arange(0, 35))),
               c='green', marker='_', lw=18, label='good/satisfactory')
    ax[1].plot(np.arange(35, 75), np.ones(len(np.arange(35, 75))),
               c='yellow', marker='_', lw=18, label='moderate')
    ax[1].plot(np.arange(75, 185), np.ones(len(np.arange(75, 185))),
               c='orange', marker='_', lw=18, label='unhealthy for \n sensitive group')
    ax[1].plot(np.arange(185, 304), np.ones(len(np.arange(185, 304))),
               c='red', marker='_', lw=18, label='unhealthy')
    ax[1].plot(np.arange(304, 604), np.ones(len(np.arange(304, 604))),
               c='purple', marker='_', lw=18, label='very unhealthy')

    ax[1].plot(np.arange(0, 100), np.zeros(len(np.arange(0, 100))),
               c='royalblue', marker='_', lw=18, label='very good')
    ax[1].plot(np.arange(100, 200), np.zeros(
        len(np.arange(100, 200))), c='green', marker='_', lw=18)
    ax[1].plot(np.arange(200, 300), np.zeros(len(np.arange(200, 300))),
               c='yellow', marker='_', lw=18)
    ax[1].plot(np.arange(300, 400), np.zeros(len(np.arange(300, 400))),
               c='orange', marker='_', lw=18)
    ax[1].plot(np.arange(400, 604), np.zeros(
        len(np.arange(400, 604))), c='red', marker='_', lw=18)
    ax[1].set_xlabel('ppb')



    # ax[1].plot(np.arange(0, 35), np.ones(len(np.arange(0, 35)))+1,
    #            c='green', marker='_', lw=18 )
    # ax[1].plot(np.arange(35, 75), np.ones(len(np.arange(35, 75)))+1,
    #            c='orange', marker='_', lw=18 )
    # ax[1].plot(np.arange(75, 185), np.ones(len(np.arange(75, 185)))+1,
    #            c='red', marker='_', lw=18 )
    # ax[1].plot(np.arange(185, 304), np.ones(len(np.arange(185, 304)))+1,
    #            c='red', marker='_', lw=18 )
    # ax[1].plot(np.arange(304, 604), np.ones(len(np.arange(304, 604)))+1,
    #            c='purple', marker='_', lw=18 )

    # --AQI label--(TH)
    ax[1].text(65, 0, '25', verticalalignment='center')
    ax[1].text(165, 0, '50', verticalalignment='center')
    ax[1].text(255, 0, '100',  verticalalignment='center')
    ax[1].text(352, 0, '200',  verticalalignment='center')
    # ax[0].text(400,0, , verticalalignment='center''323')

    # --AQI label--(US)
    ax[1].text(5, 1, '50', verticalalignment='center')
    ax[1].text(29, 1, '100', verticalalignment='center')
    ax[1].text(142, 1, '150', verticalalignment='center')
    ax[1].text(257, 1, '200', verticalalignment='center')

    ax[0].legend(bbox_to_anchor=(1.05, 1.05), frameon=False)

    labels = [item.get_text() for item in ax[0].get_yticklabels()]
    labels[0] = 'Thai AQI'
    labels[1] = 'US AQI'
   # labels[3] = 'Simplified'

    for a in ax:
        a.set_ylim([-0.5, 1.5])
        a.yaxis.set_major_locator(plt.FixedLocator([0,1]))
        a.yaxis.set_major_formatter(plt.FixedFormatter(labels))

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
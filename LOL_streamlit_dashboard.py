"""
Class:\tCS230 — Section HB1S
Name:\tYiming Zhang
\tI pledge that I have completed the programming assignment independently.
\tI have not copied the code from a student or any other source.
\tI have not given my code to any student.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print(__doc__)
print("Final Project: League of Legend Professional E-Sport Player Data Science Project Dash Board")
def aic(X,Y, k):
    y_obs = Y
    y_fitted = linear_reg.predict(X)
    residual = y_obs - y_fitted.ravel()
    SSE = sum(residual ** 2)
    AIC = 2*k - 2*np.log(SSE)
    return [X, AIC]

SELECTIONS = ['Introduction and Question of Interest', 'First Model Fit: Main Effect Model', 'Second Model Fit: Log Transformed Main Effect Model', 'Next Steps (To be continued...)']

pageSelect = st.sidebar.selectbox('Please choose a page to visit', SELECTIONS)

if pageSelect==SELECTIONS[0]:
    st.title("Final Project Dashboard: League of Legend Professional E-Sport Player Data Analytics")
    st.markdown('<p style="font-family:monospace; color:black; font-size: 12px;">Your can select a page using the sidebar on the left.</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:monospace; color:black; font-size: 12px;">Currently, you are visiting <i><b> {SELECTIONS[0]} </b></i> Page.</p>', unsafe_allow_html=True)
    st.write('In the past, when it comes to online games, people had great misconceptions and prejudices, simply equating '
             'online games with "a waste of time." However, in the 21st century, online gaming has become not only a form '
             'of entertainment but also transformed into a sporting event (i.e., e-sports). The value of e-sports players is '
             'also gradually accepted by the public. Take the phenomenal game League of Legends as an example: the top professional '
             'League of Legends players are often hyped up on the internet due to their sky-high salaries.')
    st.write("In this project, we want to examine variables that affect professional league of legend players' salaries "
             "and build a best predictive model to predict a professional their salary given his past tournament winning"
             " history, region of competition, and his/her role in a team.")
    st.markdown("The data used in this project comes from two sources: <i><a href='https://www.kaggle.com/michau96/the-best-in-league-of-legend-history'>Kaggle</a></i> "
                "and <i><a href='https://www.trackingthepros.com/PLAYERS'>TrackingThePros.com</a></i>. Both of the two "
                "sources contain data that are essential for our analysis. How we retrieved data from the two sources are "
                "different: Kaggle provides us with a cleaned dataset stored in a CSV file. "
                "On the other hand, Track The Pro is a web source that doesn't provide us with a direct download of its data. "
                "Therefore, we relied on the python package <i> Requests </i> and <i> BS4(BeautifulSoup) </i>to retrieve data from the target HTML source and eventually converted it to a CSV file using Pandas. "
                "To make the web-scrapped dataset usable, we had to do a good amount of data munging and cleaning. "
                "Eventually, we used an inner merge method to concatenate the two datasets together based on the professional "
                "players' <i> userID </i> (unique value).", unsafe_allow_html=True)
    st.markdown("In this dashboard, you would be able to see the different models we fitted to predict a professional League"
             " of Legend Player's salary and metrics about their predictive accuracy as well as model usability. You would "
             "also be able to interact with the models: use them to predict a players' salary based on your choice of input for the variables! "
             "<b> To view the description of all the variables, check the box below: </b>",  unsafe_allow_html=True)
    isDes = st.checkbox("View Description of All Variables", value=False)
    if isDes == False:
        st.write()
    else:
        st.markdown("<b> Earnings </b> : The annual salary of the professional League of Leagues player, in US dollars. ", unsafe_allow_html=True)
        st.markdown("<b> Gold     </b> : The total number of champions in the secondary (regional) League of Leagues tournament a player has already earned during his/her career. ", unsafe_allow_html=True)
        st.markdown("<b> Silver   </b> : The total number of second place in the secondary (regional) League of Leagues tournament a player has already earned during his/her career ", unsafe_allow_html=True)
        st.markdown("<b> Bronze   </b> : The total number of third place in the secondary (regional) League of Leagues tournament a player has already earned during his/her career ", unsafe_allow_html=True)
        st.markdown("<b> S-tier   </b> : The quantity of wins in the S-tier League of Leagues tournament, the world-wide annual LOL tournament, during a player’s career. ", unsafe_allow_html=True)
        st.markdown("<b> Role </b> : The role each player operates in a regular League of Leagues game. There are 6 ordinary roles: Top Laner (Top), Middle Laner (Mid), Support, ADC (attack damage carry), Jungle, and coach. ", unsafe_allow_html=True)
        st.markdown("<b> Region   </b> : The region is where their sporting clubs are located, which determines where they compete. There are several different regions: North America, China, Korea, Japan, Russia, Europe, Vietnam, PCS (Pacific Championship Series), Turkey, Brazil, Oceania, Latino America. ", unsafe_allow_html=True)

elif pageSelect==SELECTIONS[1]:
    st.markdown('<p style="font-family:monospace; color:black; font-size: 12px;">Your can select a page using the sidebar on the left.</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:monospace; color:black; font-size: 12px;">Currently, you are visiting <i><b> {SELECTIONS[1]} </b></i> Page.</p>', unsafe_allow_html=True)
    lm_Dataset = pd.read_csv('cleaned_LOL.csv')
    X = lm_Dataset[['Reg', 'Role', 'gold', 'silver', 'bronze', 's-tier']]
    X = pd.get_dummies(data=X, drop_first=True)
    Y = lm_Dataset['earnings']
    st.write("The first model we fit is the main effect model, which contains all the explanatory variables (gold, silver, bronze, s-tier, Reg and Role) and the original Y-variable (earnings in USD).")
    from sklearn.model_selection import train_test_split
    x_training, x_testing, y_training, y_testing = train_test_split(X, Y, test_size=0.2, random_state=66)
    from sklearn.linear_model import LinearRegression
    linear_reg = LinearRegression()
    lm1_skl = linear_reg.fit(x_training, y_training)
    coefficientDF = pd.DataFrame(linear_reg.coef_, X.columns, columns=['coefficients'])


    import pingouin as pg
    lm1_pg = pg.linear_regression(x_training, y_training)
    st.markdown("values for the model's <b><i> intercept </i></b>, <b><i> variable coefficients</i></b>, "
                " <b><i> coefficient significance</i></b>, <b><i> R-squared</i></b>, and <b><i>Adjusted R-squared</i>"
                "</b> can be found in the table below: ",  unsafe_allow_html=True)
    lm1_pg

    st.markdown("<b>Please use the select boxes and sliders below to adjust the inputs for the variables and predict a "
                "professional League of Legend player's salary based on the  <i> Main Effect Model </i> :</b>", unsafe_allow_html=True)
    regionList = lm_Dataset['Reg'].unique().tolist()
    regionSelect = st.selectbox("Please select the region of competition for the player: ", regionList)

    roleList = lm_Dataset['Role'].unique().tolist()
    roleSelect = st.selectbox("Please select the role in the the team for the player: ", roleList)

    goldSelect = st.slider("How many gold medal did the player get in regional tournament?", min_value= 0, max_value=int(lm_Dataset['gold'].max()), value=1, step=1)
    silvSelect = st.slider("How many silver medal did the player get in regional tournament?", min_value= 0, max_value=int(lm_Dataset['silver'].max()), value=1, step=1)
    bronzeSelect = st.slider("How many bronze medal did the player get in regional tournament?", min_value= 0, max_value=int(lm_Dataset['bronze'].max()), value=1, step=1)
    stirSelect = st.slider("How many s-tier games did the player won in world-wide tournament?", min_value= 0, max_value=int(lm_Dataset['s-tier'].max()), value=1, step=1)

    china = 0; japan = 0; euro = 0; korea = 0;  latA = 0; norAa = 0; ocean = 0; pcs =0; russia = 0; turkey = 0; viet = 0;
    coach = 0; jG = 0; mid = 0; nofix = 0; supp = 0; top = 0;


    if regionSelect == 'China':
        china = 1;
    elif regionSelect == 'Europe':
        euro=1;
    elif regionSelect == 'North America':
        norAa =1;
    elif regionSelect == 'Korea':
        korea =1;
    elif regionSelect == 'Russia':
        russia =1;
    elif regionSelect == 'Turkey':
        turkey =1;
    elif regionSelect == 'Latin America':
        latA =1;
    elif regionSelect == 'PCS':
        pcs =1;
    elif regionSelect == 'Japan':
        japan =1;
    elif regionSelect == 'Oceania':
        ocean =1;
    elif regionSelect == 'Vietnam':
        viet =1;

    if  roleSelect == 'Jungle':
        jG = 1;
    elif roleSelect == 'Top':
        top = 1;
    elif roleSelect == 'Mid':
        mid = 1;
    elif roleSelect == 'Support':
        supp = 1;
    elif roleSelect == 'No_Fixed':
        nofix= 1;
    elif roleSelect == 'Coach':
        coach = 1;

    inputDict = {'gold':goldSelect, 'silver':silvSelect, 'bronze':bronzeSelect, 's-tier':stirSelect, 'Reg_China':china , 'Reg_Europe':euro,
                 'Reg_Japan':japan, 'Reg_Korea':korea, 'Reg_Latin America':latA, 'Reg_North America':norAa, 'Reg_Oceania':ocean, 'Reg_PCS':pcs, 'Reg_Russia':russia,
                 'Reg_Turkey':turkey, 'Reg_Vietnam':viet, 'Role_Coach':coach, 'Role_Jungle':jG, 'Role_Mid':mid, 'Role_No_Fixed':nofix, 'Role_Support':supp, 'Role_Top':top}

    inputDF = pd.DataFrame(inputDict, index=[9999])
    predictedSalary = linear_reg.predict(inputDF)
    st.write(f"<p style=' font-size: 24px;'> The main effect model predicts that the {roleSelect} "
             f"players' average salary in {regionSelect}, given above merit inputs, "
             f"would be <b> ${predictedSalary[0].round(2) } </b> per year.", unsafe_allow_html=True)

    from sklearn import metrics
    import numpy as np
    st.write("")
    st.write(
        f"<p style=' font-size: 16px;'> <b> Below are some metrics to measure our model's predictive power and "
        f"overall utility </b>:",
        unsafe_allow_html=True)
    y_fittedValue = linear_reg.predict(x_testing)
    st.write("Mean_Absolute_Error|", "|    Mean Squared Error|", "|    Root Mean Squared Error|", "|      The Grand Mean")
    st.write(metrics.mean_absolute_error(y_testing, y_fittedValue), "      ",
          metrics.mean_squared_error(y_testing, y_fittedValue), "      ",
          np.sqrt(metrics.mean_squared_error(y_testing, y_fittedValue)), "        ", lm_Dataset['earnings'].mean())

    st.write("BIC:",aic(X=X, Y=Y, k=np.log(len(X)))[1])
    st.write("AIC:", aic(X=X, Y=Y, k=2)[1])

    st.write("")
    st.write(
        f"<p style=' font-size: 20px;'> <b> Model Conclusion:  </b>",
        unsafe_allow_html=True)
    st.write("This model has a very high RMSE, which is over 100% of the grand mean of the response variable, indicating that there "
             "is a huge prediction error in our model. We suspect that this could be due to skewness of the response "
             "variable. Checking the histogram of the response variable helps us to validates our guess. ")

    import matplotlib.pyplot as plt


    plt.hist(lm_Dataset['earnings'], bins=30)
    plt.title("Histogram: Earnings of LOL Pro-Players")
    plt.xlabel('Salaries')
    plt.ylabel('frequency')
    st.pyplot(plt)

    st.write("From the histogram above, it is obvious that the variable ['earnings'] is extremely right skewed. To deal"
             "with extremely right skewed outcome variables, log transformation comes in handy. On next page of the "
             "dashboard, you can find our log-transformed main effect model, and use it to predict a player's salary. ")



elif pageSelect==SELECTIONS[2]:
    st.markdown('<p style="font-family:monospace; color:black; font-size: 12px;">Your can select a page using the sidebar on the left.</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:monospace; color:black; font-size: 12px;">Currently, you are visiting <i><b> {SELECTIONS[2]} </b></i> Page.</p>', unsafe_allow_html=True)

    lm_Dataset = pd.read_csv('cleaned_LOL.csv')
    X = lm_Dataset[['Reg', 'Role', 'gold', 'silver', 'bronze', 's-tier']]
    X = pd.get_dummies(data=X, drop_first=True)
    Y = np.log(lm_Dataset['earnings'])
    st.write("The second model we fit is the log transformed main effect model, which contains all the explanatory "
             "variables (gold, silver, bronze, s-tier, Reg and Role) and the Log-transformed Y-variable "
             "(log(earnings) in USD).")
    from sklearn.model_selection import train_test_split
    x_training, x_testing, y_training, y_testing = train_test_split(X, Y, test_size=0.2, random_state=66)
    from sklearn.linear_model import LinearRegression
    linear_reg = LinearRegression()
    lm1_skl = linear_reg.fit(x_training, y_training)
    coefficientDF = pd.DataFrame(linear_reg.coef_, X.columns, columns=['coefficients'])


    import pingouin as pg
    lm1_pg = pg.linear_regression(x_training, y_training)
    st.markdown("values for the model's <b><i> intercept </i></b>, <b><i> variable coefficients</i></b>, "
                " <b><i> coefficient significance</i></b>, <b><i> R-squared</i></b>, and <b><i>Adjusted R-squared</i>"
                "</b> can be found in the table below: ",  unsafe_allow_html=True)
    lm1_pg

    st.markdown("<b>Please use the select boxes and sliders below to adjust the inputs for the variables and predict a "
                "professional League of Legend player's salary based on the  <i> Main Effect Model </i> :</b>", unsafe_allow_html=True)
    regionList = lm_Dataset['Reg'].unique().tolist()
    regionSelect = st.selectbox("Please select the region of competition for the player: ", regionList)

    roleList = lm_Dataset['Role'].unique().tolist()
    roleSelect = st.selectbox("Please select the role in the the team for the player: ", roleList)

    goldSelect = st.slider("How many gold medal did the player get in regional tournament?", min_value= 0, max_value=int(lm_Dataset['gold'].max()), value=1, step=1)
    silvSelect = st.slider("How many silver medal did the player get in regional tournament?", min_value= 0, max_value=int(lm_Dataset['silver'].max()), value=1, step=1)
    bronzeSelect = st.slider("How many bronze medal did the player get in regional tournament?", min_value= 0, max_value=int(lm_Dataset['bronze'].max()), value=1, step=1)
    stirSelect = st.slider("How many s-tier games did the player won in world-wide tournament?", min_value= 0, max_value=int(lm_Dataset['s-tier'].max()), value=1, step=1)

    china = 0; japan = 0; euro = 0; korea = 0;  latA = 0; norAa = 0; ocean = 0; pcs =0; russia = 0; turkey = 0; viet = 0;
    coach = 0; jG = 0; mid = 0; nofix = 0; supp = 0; top = 0;



    if regionSelect == 'China':
        china = 1;
    elif regionSelect == 'Europe':
        euro=1;
    elif regionSelect == 'North America':
        norAa =1;
    elif regionSelect == 'Korea':
        korea =1;
    elif regionSelect == 'Russia':
        russia =1;
    elif regionSelect == 'Turkey':
        turkey =1;
    elif regionSelect == 'Latin America':
        latA =1;
    elif regionSelect == 'PCS':
        pcs =1;
    elif regionSelect == 'Japan':
        japan =1;
    elif regionSelect == 'Oceania':
        ocean =1;
    elif regionSelect == 'Vietnam':
        viet =1;

    if  roleSelect == 'Jungle':
        jG = 1;
    elif roleSelect == 'Top':
        top = 1;
    elif roleSelect == 'Mid':
        mid = 1;
    elif roleSelect == 'Support':
        supp = 1;
    elif roleSelect == 'No_Fixed':
        nofix= 1;
    elif roleSelect == 'Coach':
        coach = 1;

    inputDict = {'gold':goldSelect, 'silver':silvSelect, 'bronze':bronzeSelect, 's-tier':stirSelect, 'Reg_China':china , 'Reg_Europe':euro,
                 'Reg_Japan':japan, 'Reg_Korea':korea, 'Reg_Latin America':latA, 'Reg_North America':norAa, 'Reg_Oceania':ocean, 'Reg_PCS':pcs, 'Reg_Russia':russia,
                 'Reg_Turkey':turkey, 'Reg_Vietnam':viet, 'Role_Coach':coach, 'Role_Jungle':jG, 'Role_Mid':mid, 'Role_No_Fixed':nofix, 'Role_Support':supp, 'Role_Top':top}

    inputDF = pd.DataFrame(inputDict, index=[9999])
    predictedSalary = linear_reg.predict(inputDF)
    st.write(f"<p style=' font-size: 24px;'> The log-transformed main effect model predicts that the {roleSelect} "
             f"players' average salary in {regionSelect}, given above merit inputs, "
             f"would be <b> ${np.exp(predictedSalary[0]).round(2) } </b> per year.", unsafe_allow_html=True)

    from sklearn import metrics
    import numpy as np
    st.write("")
    st.write(
        f"<p style=' font-size: 16px;'> <b> Below are some metrics to measure our model's predictive power and "
        f"overall utility </b>:",
        unsafe_allow_html=True)
    y_fittedValue = linear_reg.predict(x_testing)
    st.write("Mean_Absolute_Error|", "|    Mean Squared Error|", "|    Root Mean Squared Error|", "|      The Grand Mean")
    st.write(metrics.mean_absolute_error(y_testing, y_fittedValue), "      ",
          metrics.mean_squared_error(y_testing, y_fittedValue), "      ",
          np.sqrt(metrics.mean_squared_error(y_testing, y_fittedValue)), "        ", np.log(lm_Dataset['earnings']).mean())

    st.write("BIC:",aic(X=X, Y=Y, k=np.log(len(X)))[1])
    st.write("AIC:", aic(X=X, Y=Y, k=2)[1])
    import matplotlib.pyplot as plt

    st.write(
        f"<p style=' font-size: 20px;'> <b> Model Conclusion:  </b>",
        unsafe_allow_html=True)
    st.write("This model has a RMSE of 1.2516, which is about 15% of the grand mean of the response variable, indicating that there "
             "is a relatively small prediction error in our model, indicating that the log transformed main effect model"
             "is a better model at predicting League of Legend professional players' salaries. From the histogram below "
             "we can see that distribution of outcome variable (log(earning)) is approximately normal (bell-shaped).")


    plt.hist(np.log(lm_Dataset['earnings']), bins=20)
    plt.title("Histogram: Earnings of LOL Pro-Players")
    plt.xlabel('Salaries')
    plt.ylabel('frequency')
    st.pyplot(plt)
elif pageSelect==SELECTIONS[3]:
    st.markdown('<p style="font-family:monospace; color:black; font-size: 12px;">Your can select a page using the sidebar on the left.</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:monospace; color:black; font-size: 12px;">Currently, you are visiting <i><b> {SELECTIONS[3]} </b></i> Page.</p>', unsafe_allow_html=True)
    st.write("Is the log-transformed main effect model certainly the best model at predicting professional league of "
             "legend players' salaries? Absolutely not. There are a lot more potential improvements that we can try on "
             "our model: For example, we can add interaction terms to explain possible interaction effects, or power"
             "terms to explain potential non-linearity. For the next step, we would like to use stepwise variable "
             "selection method, or LASSO regression method, to achieve a model that has a better predicting power than"
             "the current ones. ")

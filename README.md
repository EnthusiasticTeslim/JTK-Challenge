# BPX-Challenge
<table>
    <tr valign=top>
        <td width="60%">
            Electric submersible pumps (ESP) are used to move high volume fluids in unconventional wells. Each pump has an average run life of 12 months and operators in the Permian Basin report that ~3 pump repairs are required weekly across their active wells. This translates to upwards of $50MM
            in expenses annually and this challenge is targeted at estimating ESP run life to improve operational efficiency.<br><br>
            We analyzed data from <b>70 wells</b> and trained machine learning models to predict precursory signals several days ahead of an ESP failure event. Additional details on the competition can be found <a href="https://www.spegcs.org/events/6836/">here</a>.
        </td>
        <td>
            <img src="ESP.png" height="auto" width="300px">
        </td>
    </tr>
</table>


### Team DJT
| Name | Affilation | Email |
| :-- | :-- | :-- |
| David Akinpelu | Louisiana State University | dakinp1@lsu.edu |
| Joses Omojola | University of Arizona | jomojo1@arizona.edu |
| Teslim Olayiwola | Louisiana State University | tolayi1@lsu.edu |


### Environment setup
Python virtual environments are used for this project. Execute the commands below in terminal to install all requirements.
```bash
~$chmod +x setup.sh
~$sh setup.sh
~$source jtk/bin/activate
```
Documentation for running the different scripts can be found [here](Documentation.md). <br> <br>

Link to the presentation slides can be found [here](https://docs.google.com/presentation/d/1-Hj4dXrPa_KKDlAl1cg9mzSkDlehT44X).


### To-Do
- [x] Complete data cleaning for spikes
- [x] Resample data for cropping
- [x] Crop timeseries for training model
- [x] Handle nan values
- [x] Create model architecture with LSTM
- [x] Improve model architecture - (Limited time for multiple tests)

import urllib.request
import deeplabcut
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style


def install_dependencies():
    """
    Install the required dependencies for the code.
    """
    dependencies = ['wxPython==4.0.7.post2', 'tensorflow==1.14', 'ffmpeg', 'deeplabcut']
    for dependency in dependencies:
        os.system(f"pip install {dependency} --user")


def download_dlc_cpu():
    """
    Download the DLC-CPU file manually for CPU usage.
    """
    print('Beginning DLC-CPU download')
    url = 'http://www.mousemotorlab.org/s/DLC-CPU.yaml'  # Download address
    destination = r'C:/Users/ariel/desktop/DLC-CPU.YAML'  # Destination for the download is the desktop
    urllib.request.urlretrieve(url, destination)


def prepare_files():
    """
    Prepare the files for analyzing data.
    """
    video_path = 'C:/Users/ariel/Desktop/hornets'
    project_path = 'C:/Users/ariel'

    # Create a new project
    deeplabcut.create_new_project('Hornets', 'Ariel', [video_path],
                                  working_directory=project_path,
                                  copy_videos=True)

    # Open the config file for editing
    config_file_path = os.path.join(project_path, 'hornet', 'config2.yaml')
    os.startfile(config_file_path)

    # Extract frames for labeling
    deeplabcut.extract_frames(config_file_path, 'automatic', 'kmeans')

    # Label the frames
    deeplabcut.label_frames(config_file_path)

    # Check the labeled frames
    deeplabcut.check_labels(config_file_path)


def train_program():
    """
    Train the program using the labeled frames.
    """
    config_file_path = 'C:/Users/ariel/Desktop/hornet/config2.yaml'

    # Create the training dataset
    deeplabcut.create_training_dataset(config_file_path)

    # Train the network
    deeplabcut.train_network(config_file_path)

    # Evaluate the network
    deeplabcut.evaluate_network(config_file_path, Shuffles=[1], plotting=True)


def analyze_videos():
    """
    Analyze the videos using the trained network.
    """
    config_file_path = 'C:/Users/ariel/Desktop/hornet/config2.yaml'
    video_path = 'C:/Users/ariel/eating shrew-Ariel-2020-06-07/videos/eating shrew.mp4'

    # Analyze the videos and save as CSV
    deeplabcut.analyze_videos(config_file_path, [video_path], save_as_csv=True)

    # Plot the trajectories
    deeplabcut.plot_trajectories(config_file_path, [video_path], filtered=True)

    # Create a labeled video
    deeplabcut.create_labeled_video(config_file_path, [video_path], filtered=True)


def data_analysis():
    """
    Perform data analysis on the analyzed videos.
    """
    headered_data = r'C:\Users\ariel\eating shrew-Ariel-2020-06-07\videos\eating shrewDLC_resnet50_eating shrewJul7shuffle1_18000.csv'
    headerskip(headered_data)

    shrew_data = "data_without_header.csv"
    data = pd.read_csv(shrew_data, header=[0, 1], index_col=0)
    data.columns = data.columns.map('_'.join)
    data.rename_axis('bodyparts-coords')

    # Drop the likelihood columns
    cleandata = data.drop(['Nosetip_likelihood', 'rightwhiskers_likelihood', 'leftwhiskers_likelihood',
                           'rightear_likelihood', 'leftear_likelihood'], axis=1)
    cleandata = cleandata.iloc[:350, ]

    def calculatelocation(x, y):
        """
        Calculate the relative location based on Pythagorean theorem.
        """
        location = []
        for xstat in x:
            for ystat in y:
                location.append(math.sqrt(xstat ** 2) + (ystat ** 2))
        return location

    # Calculate the location data for each body part
    cleandata = cleandata.join(pd.DataFrame({
        'Nosetip_location': calculatelocation(cleandata['Nosetip_x'], cleandata['Nosetip_y']),
        'rightwhiskers_location': calculatelocation(cleandata['rightwhiskers_x'], cleandata['rightwhiskers_y']),
        'leftwhiskers_location': calculatelocation(cleandata['leftwhiskers_x'], cleandata['leftwhiskers_y']),
        'rightear_location': calculatelocation(cleandata['rightear_x'], cleandata['rightear_y']),
        'leftear_location': calculatelocation(cleandata['leftear_x'], cleandata['leftear_y'])
    }, index=cleandata.index))

    # Combine all the data for each body part under its header
    final_shrew_data = cleandata.sort_index(axis=1)

    return final_shrew_data


def plot_data(final_shrew_data):
    """
    Plot the data and perform analysis.
    """
    # Plot the location of body parts in time
    ax = plt.gca()
    final_shrew_data.plot(kind='line', y='Nosetip_location', color='black', ax=ax)
    final_shrew_data.plot(kind='line', y='rightwhiskers_location', color='red', ax=ax)
    final_shrew_data.plot(kind='line', y='leftwhiskers_location', color='yellow', ax=ax)
    final_shrew_data.plot(kind='line', y='rightear_location', color='green', ax=ax)
    final_shrew_data.plot(kind='line', y='leftear_location', color='blue', ax=ax)
    plt.xlabel("Time (frames)")
    plt.ylabel("Coordinates")
    plt.title('Body parts location in time')
    plt.show()

    # Calculate the delta coordinates of body parts between frames
    deltas = final_shrew_data.diff(periods=-1)
    deltas = deltas.dropna()

    # Calculate the total movement of each body part
    delta_data = {
        'Nosetipmovement': calculatelocation(deltas['Nosetip_x'], deltas['Nosetip_y']),
        'rightwhiskersmovement': calculatelocation(deltas['rightwhiskers_x'], deltas['rightwhiskers_y']),
        'leftwhiskersmovement': calculatelocation(deltas['leftwhiskers_x'], deltas['leftwhiskers_y']),
        'rightearmovement': calculatelocation(deltas['rightear_x'], deltas['rightear_y']),
        'leftearmovement': calculatelocation(deltas['leftear_x'], deltas['leftear_y'])
    }

    for key in delta_data:
        delta_data[key] = sum(delta_data[key])

    # Plot the total movement of each body part
    names = ['Nose tip', 'right whiskers', 'left whiskers', 'right ear', 'left ear']
    values = list(delta_data.values())
    plt.bar(range(len(names)), values, tick_label=names)
    plt.xlabel("Shrew body parts")
    plt.ylabel("Total movement")
    plt.title('Body parts movement')
    plt.show()

    # Plot the 3D movement of the Nosetip
    style.use('ggplot')
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    x = final_shrew_data['Nosetip_x']
    y = final_shrew_data['Nosetip_y']
    z = final_shrew_data['Nosetip_location']
    ax1.scatter(x, y, z, c='m', marker='o')
    ax1.set_xlabel('X_coordinates')
    ax1.set_ylabel('Y_coordinates')
    ax1.set_zlabel('Z_coordinates')
    ax1.set_title('Nose tip 3D movement')
    plt.show()


# Main function to execute the code
def main():
    install_dependencies()
    download_dlc_cpu()
    prepare_files()
    train_program()
    analyze_videos()
    final_shrew_data = data_analysis()
    plot_data(final_shrew_data)


# Run the main function
if __name__ == "__main__":
    main()

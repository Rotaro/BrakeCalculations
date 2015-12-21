import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import lines as mlines
import matplotlib.animation as animation
                
class BrakeTest:
    """
    Generates brake test data. 
    
    Brake pressure and brake force are generated as linear functions of 
    measurement time. Random noise is added.

    Arguments:
    n_test      - Number of data points to generate.
    """

    #Max values for brake pressure and force, based on real tests.
    MAX_PRESS = 3.7
    MAX_FORCE = 45
    
    def __init__(self, n_test):
        """
        n_roll      - Number of data points used to measure rolling resistance.
        roll_res    - Rolling resistance.
        press_k     - Linear dependence of brake pressure on measurement time.
        press_std   - Standard deviation of Gaussian random noise in brake presure.
        force_k     - Linear dependence of brake force on measurement time.
        force_std   - Standard deviation of Gaussian random noise in brake force.
        force_delay - Number of data points until brake force starts rising (will 
                      determine wake-up pressure).
        """
        self.n_test = n_test
        #set braking characteristics 
        self.n_roll = round(self.n_test/6)
        self.force_delay = round(self.n_roll/2)
        self.press_std = 0.01
        self.force_std = 0.1
        
        #randomly generate braking characteristics
        #press_k is chosen to have wake-up pressure between 0.2 and 1 bar, 
        #and also to generate 3.7 pressure by end of test
        min_k = 3.7/(self.n_test-self.n_roll-1)
        max_k = 1.0/self.force_delay
        self.press_k = min_k + np.random.rand()*(max_k-min_k)
        self.force_k = self.press_k*(3+np.random.rand()*13)
        self.roll_res = 2+np.random.rand()*8
        

    def generate_data(self):
        """
        Generates brake test data.
        """
        self.t_axis = np.arange(0, self.n_test)
        
        #Generate basic data
        self.press_axis = np.zeros(self.n_test)
        self.press_axis[self.n_roll:] = \
            self.press_k*(self.t_axis[self.n_roll:]-self.t_axis[self.n_roll])
        self.force_axis = np.zeros(self.n_test) + self.roll_res
        self.force_axis[self.n_roll+self.force_delay:] = \
            self.roll_res + self.force_k*(self.t_axis[self.n_roll+self.force_delay:]
                                          -self.t_axis[self.n_roll+self.force_delay])     
        #Cut off if MAX values are breached
        self.t_cutoff = self.n_test
        press_lim = np.size(self.press_axis[self.press_axis < self.MAX_PRESS])
        force_lim = np.size(self.force_axis[self.force_axis < self.MAX_FORCE])
        if (press_lim <= force_lim):
            if (press_lim != self.n_test):
                self.t_cutoff = press_lim
                self.press_axis[press_lim:] = self.press_axis[press_lim]
                self.force_axis[press_lim:] = self.force_axis[press_lim]
        else:
            if (force_lim != self.n_test):
                self.t_cutoff = force_lim
                self.press_axis[force_lim:] = self.press_axis[force_lim]
                self.force_axis[force_lim:] = self.force_axis[force_lim]
        #Generate noise
        self.press_axis[self.n_roll:] = \
            self.press_axis[self.n_roll:] + np.sqrt(self.press_axis[self.n_roll:])* \
            np.random.normal(0.0, self.press_std, self.n_test-self.n_roll)
        self.force_axis = \
            self.force_axis + np.sqrt(self.force_axis)* \
            np.random.normal(0.0, self.force_std, self.n_test)

class BrakePlotting:
    """
    Calculates and plots (with animations!) the brake test results. 
    The same plot is used throughout the run.
    """
    def __init__(self, brake_test):
        self.brkt = brake_test

    def line_from_pts(self, x0, y0, x1, y1):
        """
        Returns coefficients for a straight line between two points.
        """
        a = (y1-y0)/(x1-x0)
        b = y0-a*x0
        return [a, b]   

    def create_plot(self):
        """
        Creates an empty plot and stores it in object variables.
        """
        fig1, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        self.fig = fig1
        self.axs = [ax1, ax2]
        self.status = 1
        cid = fig1.canvas.mpl_connect('key_press_event', self.advance_plot)
        ax1.set_title("Press any key to start animation.")
        plt.show()

    def reset_plot(self):
        """
        Cleans axes of object's plot.
        """
        try:
            if (len(self.axs) > 0):
                self.axs[0].cla()
                self.axs[1].cla()
            if (self.status != 1):
                self.axs[1].set_visible(False)
            else:
                self.axs[1].set_visible(True)
        except NameError:
            print("No plot to reset")

    def advance_plot(self, event):
        """
        Updates plot to next stage. Plugged into event handler 
        for key presses in plot window.
        """
        self.cease_anim = 1
        self.reset_plot()
        if self.status == 1:
            self.status = 2
            self.plot_raw_data()
        elif self.status == 2:
            self.status = 3
            self.plot_calc_6bar()
        elif self.status == 3:
            self.status = 4
            self.plot_calc_z()
        elif self.status == 4:
            self.status = 1

    def plot_raw_data(self):
        """
        Plots raw test data.
       
        Also plots lines for rolling resistance and wake-up pressure.
        """
        ax1 = self.axs[0]
        ax2 = self.axs[1]
        #Axes
        ax1.set_title("Brake pressure and force measurement")
        ax1.set_ylim([0, 6])
        ax1.set_xlabel("time (a.u.)")
        ax1.set_ylabel("Pressure (bar)", color="b")
        for tick in ax1.get_yticklabels():
            tick.set_color("b")
        ax2.set_ylim([0, np.max(self.brkt.force_axis)+5])
        ax2.set_ylabel("Force (kN)", color="r")
        for tick in ax2.get_yticklabels():
            tick.set_color("r")
        ax1.set_xlim([0, np.max(self.brkt.t_axis)])
        ax2.set_xlim([0, np.max(self.brkt.t_axis)])
        #Animation
        self.cease_anim = 0
        plt1, = ax1.plot(self.brkt.t_axis[0], 
                        self.brkt.press_axis[0], "b-", label="Brake Pressure") 
        plt2, = ax2.plot(self.brkt.t_axis[0], 
                        self.brkt.force_axis[1], "r-", label="Brake Force")
        comb_data = np.array([self.brkt.t_axis, self.brkt.press_axis, self.brkt.force_axis])

        def update_lines(num):
            if (self.status != 2):
                raise StopIteration
            plt1.set_data(comb_data[[0,1],:num])
            plt2.set_data(comb_data[[0,2],:num])
            if (num == np.size(self.brkt.t_axis)-3):
                ax1.axvline(self.brkt.t_axis[self.brkt.n_roll], 0, 
                            max(ax1.get_yticks()), color='black', ls='--')
                ax1.axvline(self.brkt.t_axis[self.brkt.t_cutoff], 0, 
                            max(ax1.get_yticks()), color='black', ls='--')
                ax1.axvline(self.brkt.t_axis[self.brkt.n_roll+self.brkt.force_delay], 0,
                            max(ax1.get_yticks()), color='black', ls='--')
                ax2.axhline(self.brkt.roll_res, 0, 
                            max(ax1.get_xticks()), color='black', ls='--')
                ax1.axhline(self.brkt.press_axis[self.brkt.n_roll+self.brkt.force_delay],
                            0, max(ax1.get_xticks()), color='black', ls='--')
                ax2.text(self.brkt.t_axis[(self.brkt.t_cutoff+self.brkt.n_roll)/2], 
                         self.brkt.roll_res+np.max(ax2.get_yticks())/60, 
                         "Rolling resistance $F_{rd}$")
                ax1.text(self.brkt.t_axis[(self.brkt.t_cutoff+self.brkt.n_roll)/2], 
                         self.brkt.press_axis[self.brkt.n_roll+self.brkt.force_delay]-0.3,
                         "Wake-up pressure $p_w$")
                ax1.set_title(
                    "Brake pressure and force measurement\nPress any key to continue.")
            return plt1,
        #try-except is used to stop the animation if the user moves on
        try:
            animation.FuncAnimation(self.fig, update_lines, np.size(self.brkt.t_axis)-2, 
                               interval=100, repeat=False)
        except StopIteration:
            print("Animation stopped.")

        #Legend
        ax1.legend([plt1, plt2], [l.get_label() for l in [plt1, plt2]], loc=2)
        plt.show()

    def plot_calc_6bar(self):
        """
        Calculates and plots extrapolation of brake force to 6 bars.
        
        Removes rolling resistance from the brake force, and extrapolates 
        the brake force at 6 bar brake pressure using linear regression.
        """
        #Sort data according to brake pressure
        rel_data = self.brkt.n_roll+self.brkt.force_delay
        press_ind = np.argsort(self.brkt.press_axis[rel_data:self.brkt.t_cutoff])
        press_sort = (self.brkt.press_axis[rel_data:self.brkt.t_cutoff])[press_ind]
        force_sort = (self.brkt.force_axis[rel_data:self.brkt.t_cutoff])[press_ind]
        #Fit simple line to data
        self.lin_fit = np.polyfit(press_sort, force_sort-self.brkt.roll_res, 1)
        lin_fit_x = np.linspace(0,6,100)
        lin_fit_y = np.polyval(self.lin_fit, lin_fit_x)
        #Configure plot axes
        self.axs[0].set_xlim([0,6])
        self.axs[0].set_title("Brake pressure vs. brake force")
        self.axs[0].set_xlabel("Pressure (bar)", color="b")
        self.axs[0].set_ylabel("Force (kN)", color="r")
        self.axs[0].set_ylim([0, np.max(force_sort)+10])
        self.cease_anim = 0
        #Animation
        plt1, = self.axs[0].plot(press_sort[0], force_sort[0]-self.brkt.roll_res, 
                                label="Brake Force - Rolling Resistance", color="r")
        comb_data = np.array([press_sort, force_sort-self.brkt.roll_res])
        def update_lines(num):
            if (self.status != 3):
                raise StopIteration
            plt1.set_data(comb_data[...,:num])
            if (num == np.size(press_sort)-3):
                plt2, = self.axs[0].plot(lin_fit_x, lin_fit_y, 
                                label="Line fit (least squares)", color="g")
                self.axs[0].legend([plt1,plt2],
                   [l.get_label() for l in [plt1,plt2]], loc=2)
                self.axs[0].text(press_sort[np.size(press_sort)/2]+1, 
                                 np.average(lin_fit_y)-2,
                                 "%.2f $p$ %.2f" % tuple(self.lin_fit))
                self.axs[0].set_title(
                    "Brake pressure vs. brake force\nPress any key to continue.")
            return plt1,
        try:
            animation.FuncAnimation(self.fig, update_lines, np.size(force_sort)-2, 
                               interval=100, repeat=False)
        except StopIteration:
            print("Animation stopped.")
        
        plt.show()
        
    def plot_calc_z(self):
        """
        Calculates and plots z, i.e. proportion of brake force to axle weight. 
       
        Also draws brake corridors (i.e. acceptable values for z).
        """
        #Sort data according to brake pressure
        rel_data = self.brkt.n_roll+self.brkt.force_delay
        press_ind = np.argsort(self.brkt.press_axis[rel_data:])
        press_sort = (self.brkt.press_axis[rel_data:])[press_ind]
        force_sort = (self.brkt.force_axis[rel_data:])[press_ind]
        #Calculate z (mass is unknown, so z is simply forced to be 0.8 at 6 bar)
        lin_fit_x = np.linspace(0,6,4)
        lin_fit_y = np.polyval(self.lin_fit, lin_fit_x)
        z_fit = lin_fit_y/np.max(lin_fit_y)*0.8
        #Configure plot axes
        self.axs[0].set_xlim([0,8])
        self.axs[0].set_xlabel("Pressure (bar)", color="b")
        self.axs[0].set_ylabel("z", color="k")
        self.axs[0].set_ylim([0, 0.8])
        self.axs[0].set_title("Brake pressure vs. z")
        #Animation 
        unl = Rectangle((0, 0), 1, 1, fc="red") #proxy for unloaded colored area
        l = Rectangle((0, 0), 1, 1, fc="green") #proxy for loaded colored area 
        fit = Rectangle((0, 0), 1, 1, fc="magenta") #proxy for loaded colored area 
        fit = mlines.Line2D([], [], color='magenta')
        second_time = 0 #prevents animation from calling same function twice..
        def update_lines(num):
            if (self.status != 4):
                raise StopIteration
            if (num == 0 and second_time == 1):
                #Unloaded corridor
                unloaded_1 = self.line_from_pts(0.2, 0, 5.5, 0.8)
                self.axs[0].plot([1.0, 4.5, 7.5], 
                                 np.array([1.0, 4.5, 7.5])* \
                                     unloaded_1[0]+unloaded_1[1], "r-", 
                                 label="Unloaded")
                self.axs[0].plot([1.0, 4.5, 7.5], [0, 0.4, 0.65], "r-")
                self.axs[0].fill_between([0, 1.0, 4.5, 7.5], [0, 0, 0.4, 0.65], 
                                 np.array([0, 1.0, 4.5, 7.5])* \
                                     unloaded_1[0]+unloaded_1[1], 
                                 facecolor='red', alpha=0.5, label="Unloaded")
                self.axs[0].legend([unl], ["Unloaded"], loc=2)
            elif (num == 1):
                #Loaded corridor
                loaded_1 = self.line_from_pts(0.2, 0, 7.5, 0.8)
                self.axs[0].plot([1.0, 4.5, 7.5], 
                                 np.array([1.0, 4.5, 7.5])* \
                                     loaded_1[0]+loaded_1[1], "g-",
                                 label="Loaded")
                self.axs[0].plot([1.0, 4.5, 7.5], [0, 0.35, 0.575], "g-")
                self.axs[0].fill_between([0, 1.0, 4.5, 7.5], [0, 0, 0.35, 0.575], 
                                 np.array([0, 1.0, 4.5, 7.5])* \
                                     loaded_1[0]+loaded_1[1], 
                                 facecolor='green', alpha=0.5, label='Loaded')
                self.axs[0].legend([unl, l], ["Unloaded", "Loaded"], loc=2)
            elif (num == 7): #not used at the moment
                #Vertical/horizontal lines to show corridor edges
                self.axs[0].axvline(4.5, 0, 0.4/0.8, color='black', ls='--')
                self.axs[0].axvline(5.5, 0, 0.8/0.8, color='black', ls='--')
                self.axs[0].axvline(7.5, 0, 0.8/0.8, color='black', ls='--')
                self.axs[0].axhline(0.1, 0, 2.0/8, color='black', ls='--')
                self.axs[0].axhline(0.35, 0, 4.5/8, color='black', ls='--')
                self.axs[0].axhline(0.40, 0, 4.5/8, color='black', ls='--')
                self.axs[0].axhline(0.575, 0, 7.5/8, color='black', ls='--')
                self.axs[0].axhline(0.650, 0, 7.5/8, color='black', ls='--')
            elif (num == 3):
                #Linear fit
                self.axs[0].plot(lin_fit_x, z_fit, 
                                        label="Line fit (least squares)", color="m")
                self.axs[0].legend([unl, l, fit], 
                                   ["Unloaded", 
                                    "Loaded", 
                                    "Line fit (least squares)"], loc=2)
                self.axs[0].set_title(
                    "Brake pressure vs. z\nPress any key to restart animation.")
        try:
            animation.FuncAnimation(self.fig, update_lines, 4,
                               interval=1500, repeat=False)
        except StopIteration:
            print("Animation stopped.")
        second_time = 1
        plt.show()


        
if (__name__ == "__main__"):
    brkt = BrakeTest(60)
    brkt.generate_data()
    brkp = BrakePlotting(brkt)
    brkp.create_plot()

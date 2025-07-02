from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import special
import random as rn
import math
import time
from typing import List,Tuple,TypeVar
import argparse
import os
from tqdm import tqdm

cur_dir = os.getcwd()

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Dataset1_gold_0.25.xlsx", help="Path to dataset .xlsx file")
    parser.add_argument("--grid_resolution", type=int, default=80)
    parser.add_argument("--output_dir", type=str, default="datasets/results", help="Path where grids are stored")
    parser.add_argument("--cell_constant", type=float, default=4.0701105)
    parser.add_argument("--n_cells", type=int, default=125)
    parser.add_argument("--solid_fraction", type=float, default=0.25)
    parser.add_argument("--crystal_system", type=str, default='FCC')
    parser.add_argument("--keep_lmp_files", action="store_true")
    return vars(parser.parse_args())

def read_lmp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_atoms = int(lines[1].strip().split()[0])
    x_bounds = list(map(float, lines[3].strip().split()[:2]))
    y_bounds = list(map(float, lines[4].strip().split()[:2]))
    z_bounds = list(map(float, lines[5].strip().split()[:2]))
    
    atoms = []
    atom_section = False
    for line in lines:
        if line.strip() == 'Atoms # atomic':
            atom_section = True
            continue
        if atom_section:
            if line.strip():
                atoms.append(list(map(float, line.strip().split()[2:])))
    
    return np.array(atoms), x_bounds, y_bounds, z_bounds

def create_binary_grid(atoms, x_bounds, y_bounds, z_bounds, resolution):
    x_len = x_bounds[1] - x_bounds[0]
    y_len = y_bounds[1] - y_bounds[0]
    z_len = z_bounds[1] - z_bounds[0]
    
    grid = np.zeros((resolution, resolution, resolution), dtype=int)
    
    x_factor = resolution / x_len
    y_factor = resolution / y_len
    z_factor = resolution / z_len
    
    for atom in atoms:
        x, y, z = atom
        x_idx = int((x - x_bounds[0]) * x_factor)
        y_idx = int((y - y_bounds[0]) * y_factor)
        z_idx = int((z - z_bounds[0]) * z_factor)
        
        if x_idx < resolution and y_idx < resolution and z_idx < resolution:
            grid[x_idx, y_idx, z_idx] = 1
    
    return grid

class Nano_Porous_Material():
    def __init__(self,atoms_coordinates : List[List[float]],box_size : List[float], random_waves_phases: List[float],origin : List[float]=[0,0,0]) -> None:
        """
        Constructor:\n
        * Nano_Porous_Material(atoms_coordinates,box_size, random_waves_phases,origin=[0,0,0])\n
            Arguments:\n
                atoms_coordinates (list[list[float]]) : Atoms Coordinations Matrix of the Nano Porous Material (shape - [N,3]).\n
                box_size (list[float]): Box Size Array [x,y,z].\n
                random_wave_phases (list[float]) : Array of the Random Wave Phases of the Nano Porous Material.\n
                origin (list[float]): Origin Point [x0,y0,z0].\n
        """
        self._atoms_coordinates=atoms_coordinates
        self._box_size=box_size
        self._random_waves_phases=random_waves_phases
        self.origin=origin
    
    def get_Atoms_Coordinates_Matrix(self)->List[List[float]]:
        """
        Returns:
            Matrix [Number of Atoms by 3] of XYZ Coordinates
        """
        return np.copy(self._atoms_coordinates)
    
    def get_Random_Waves_Phases(self)->List[float]:
        """
        Returns:
            Random Wave Phases (Phi's)
        """
        if self._random_waves_phases is None:
            return None
        return np.copy(self._random_waves_phases)

    def get_Box_Size(self)->List[float]:
        """
        Returns:
            Box Size of the Nano Porous Material [x,y,z]
        """
        return np.copy(self._box_size)

    def get_Origin(self)->List[List[float]]:
        """
        Returns:
            Origin Point of the Nano Porous Material [x0,y0,z0]
        """
        return np.copy(self.origin)

    def create_XYZ_File(self,name : str, path : str="")->None:
        """
        Creating XYZ File of the Nano Porous Material [name.txt]\n
        Arguments:\n
            coordinate_matrix (list[float]): Matrix of (x,y,z) Positions of Atoms\n
            name (string): The Name of the Created XYZ File\n
            path (string,optional): The Path of the New File (defualt: The Path of the Code File)\n
        """
        if path!="":
            file_path=f"{path}/{name}.txt"
        else:
            file_path=f"{name}.txt"
        with open(file_path,'w') as f:
            f.write("%s\n" % len(self._atoms_coordinates))
            f.write(f"Lattice=\"{self._box_size[0]} 0.0 0.0 0.0 {self._box_size[1]} 0.0 0.0 0.0 {self._box_size[2]}\" Properties=pos:R:3")
            for i in range(0,len(self._atoms_coordinates)):
                f.write("\n%s %s %s" %(self._atoms_coordinates[i][0],self._atoms_coordinates[i][1],self._atoms_coordinates[i][2]))

    def create_Lammps_File(self,name : str,path : str="")->None:
        """
        Creating Lammps File of the Nano Porous Material [name.txt]\n
        Arguments:\n
            name: The Name of the Created XYZ File\n
            path (string,optional): The Path of the New File (defualt: The Path of the Code File)
        """
        if path!="":
            file_path=f"{path}/{name}.txt"
        else:
            file_path=f"{name}.txt"
        box_size=self.get_Box_Size()
        i=1
        with open(file_path,'w') as f:
            f.write(f"#LAMMPS Data File\n {len(self._atoms_coordinates[:,0])} atoms\n1 atom types\n0 {box_size[0]} xlo xhi\n0 {box_size[1]} ylo yhi\n0 {box_size[2]} zlo zhi\n\nAtoms # atomic\n")
            for coor in self._atoms_coordinates:
                f.write(f"\n{i} 1 {coor[0]} {coor[1]} {coor[2]}")
                i+=1
    
    def create_Random_Phases_File(self,name : str,path : str="")-> None:
        """ 
        Creating txt File of the Random Phases\n
        Arguments:\n
            name: The Name of the Random Phases File\n
            path (string,optional): The Path of the New File (defualt: The Path of the Code File)
        """
        file_path=f"{path}/{name}.txt"
        i=1
        with open(file_path, 'w') as f:
            for rand_phi in self.get_Random_Waves_Phases():
                f.write(f"{i}\t{rand_phi}\n")
                i+=1

class Nano_Porous_Material_Generator():
    def __init__(self, a : float, H_2 : int, R : int, SV : float,random_wave_phases_array : List[float]=None, crystal_system : str='FCC') -> None:
        """
        Constructor: Create Nano_Pouros_Generator Object
        Arguments:
            a (float): Basis Length 
            H_2 (int): (Wave Direction Number)^2
            R (int): Basis Repititions
            SV (float): Solid Volume Fraction
            random_wave_phases_array (list[float], optional): Array of the Random Wave Phases. (Defaults: None for Randomized Wave Phases Array)
            crystal_system (str, optional): Cells' crystal system ['FCC'/'BCC'/'SC']. (Defaults: 'FCC')
        """
        if not (isinstance(a, (int, float))):
            raise ValueError(f"Invalid Constructor: 'a' must be a number")
        if not (isinstance(H_2, int)):
            raise ValueError(f"Invalid Constructor: 'H_2' must be an Integer")
        if not (isinstance(R, int)):
            raise ValueError(f"Invalid Constructor: 'R' must be an Integer")
        if not (isinstance(SV, (int, float))):
            raise ValueError(f"Invalid Constructor: 'SV' must be a number")
        if not (isinstance(crystal_system,str)):
            raise ValueError(f"Invalid Constructor: 'crystal_system' must be a string")
        if ((SV>1) | (SV<0)):
            raise ValueError(f"Invalid Constructor: 'SV' must be a number between 0 to 1")
        if ((crystal_system !='FCC') & (crystal_system !='BCC') & (crystal_system !='SC')):
            raise ValueError(f"Invalid Constructor: 'crystal_system' must be of the value of -> 'FCC','BCC' or 'SC' ")  
        self.a=a
        self.h_2=H_2
        self.r=R
        self.sv=SV
        self.random_waves_phases=random_wave_phases_array
        self.crystal_system=crystal_system

    def generate(self)->Nano_Porous_Material:
        """
        Generating a New Nano Porous Material
        Returns:
            New Nano Porous Material
        """
        box_length=self.a*self.r
        old_random_wave=self.get_Random_Waves_Phases()
        atoms_coordinates=self._generate_Nano_Porous_Material()
        npm=Nano_Porous_Material(atoms_coordinates,[box_length,box_length,box_length],self.get_Random_Waves_Phases())
        self.random_waves_phases=self.set_Random_Wave_Phases(old_random_wave)
        return npm
    
    # get Functions    
    def get_Atoms_Coordinates_Matrix(self)->List[List[float]]:
        """
        Returns:
            Matrix [Number of Atoms by 3] of XYZ Coordinates
        """
        return np.copy(self.atoms_coordinates)

    def get_a(self)->float:
        """
        Returns:
            Basis Length (a)
        """
        return np.copy(self.a)
    
    def get_R(self)->float:
        """
        Returns:
            Basis Repititions (R)
        """
        return np.copy(self.r)
    
    def get_H_2(self)->float:
        """
        Returns:
           (Wave Direction Number)^2 (H^2)
        """
        return np.copy(self.h_2)
    
    def get_SV(self)->float:
        """
        Returns:
            Solid Volume Fraction (SV)
        """
        return np.copy(self.h_2)
    
    def get_Crystal_System(self)->str:
        """
        Returns:
            Cells' Crystal System
        """
        return np.copy(self.crystal_system)
    
    def get_Random_Waves_Phases(self)->List[float]:
        """
        Returns:
            Random Wave Phases (Phi's)
        """
        if self.random_waves_phases is None:
            return None
        return np.copy(self.random_waves_phases)
    
    # set Functions
    def set_a(self,a :float)->None:
        """
        Changing the Basis Length of the Nano Porous Generated Material
        Arguments:
            a (float): Basis Length 
        """
        if not (isinstance(a, (int, float))):
            raise ValueError(f"Invalid Constructor: 'a' must be a number")
        self.a=a
    
    def set_R(self,R :int)->None:
        """
        Changing the Basis Repititions of the Nano Porous Generated Material
        Arguments:
            R (int): Basis Repititions
        """
        if not (isinstance(R, int)):
            raise ValueError(f"Invalid Constructor: 'R' must be an Integer")
        self.r=R
    
    def set_H_2(self,H_2 :int)->None:
        """
        Changing the (Wave Direction Number)^2 of the Nano Porous Generated Material
        Arguments:
            H_2 (int): (Wave Direction Number)^2
        """
        if not (isinstance(H_2, int)):
            raise ValueError(f"Invalid Constructor: 'H_2' must be an Integer")
        self.h_2=H_2
        
    def set_SV(self,SV :float)->None:
        """
        Changing the Solid Volume Fraction of the Nano Porous Generated Material
        Arguments:
            SV (float): Solid Volume Fraction 
        """
        if not (isinstance(SV, (int, float))):
            raise ValueError(f"Invalid Constructor: 'SV' must be a number")
        if ((SV>1) | (SV<0)):
            raise ValueError(f"Invalid Constructor: 'SV' must be a number between 0 to 1")
        self.sv=SV
        
    def set_Crystal_System(self,crystal_system :str)->None:
        """
        Changing the Cells' Crystal System of the Nano Porous Generated Material
        Arguments:
            crystal_system (float): Cells' Crystal System
        """
        if not (isinstance(crystal_system,str)):
            raise ValueError(f"Invalid Constructor: 'crystal_system' must be a string")
        if ((crystal_system !='FCC') & (crystal_system !='BCC') & (crystal_system !='SC')):
            raise ValueError(f"Invalid Constructor: 'crystal_system' must be of the value of ->  'FCC','BCC' or 'SC' ")
        self.crystal_system=crystal_system
    
    def set_Random_Wave_Phases(self,random_wave_phases_array : List[float]):
        """
        Changing the Array of the Random Wave Phases of the Nano Porous Generated Material
        Arguments:
            random_wave_phases_array (list[float]): Array of the Random Wave Phases. Use None for Randomized Wave Phases Array
        """   
        self.random_waves_phases=random_wave_phases_array
        
    # Generate Nano Porous Material:
    def _generate_Nano_Porous_Material(self)->List[List[float]]:
        """
        Generating a Random Nano Porous Material
        """
        # Getting Needed Properties
        a=self.a
        R=self.r
        SV=self.sv
        # Generate Direction Waves
        waves_direc=self._generate_Waves()
        n=len(waves_direc[:,0])
        # Simulation Properties
        L=R*a
        ID=2*math.pi*waves_direc/L
        epsilon=np.emath.sqrt(2)*special.erfinv(2*SV-1)
        # Generate Random Phi List
        if self.random_waves_phases is None:
            phi=self._generate_List_of_Random_Phi(n)
            self.random_waves_phases=phi
        elif len(self.random_waves_phases)!=n:
            raise ValueError("The Number of Input Waves Doesn't Correspound With Selected H")
        else:
            phi=self.random_waves_phases
        crystal_system=self.get_Crystal_System()
        # FCC Basis
        if (crystal_system=='FCC'):
            basis=np.array([[0,0,0],\
                            [a/2, a/2, 0],\
                            [a/2, 0, a/2],\
                            [0, a/2, a/2]])
            b_num=4
        # BCC Basis
        if (crystal_system=='BCC'):
            basis=np.array([[0,0,0],\
                            [a/2, a/2, a/2]])
            b_num=2
        # SC Basis
        if (crystal_system=='SC'):
            basis=np.array([[0,0,0]])
            b_num=1
        # Create Arrays to Work with
        count=0
        f=np.zeros(R**3*b_num)
        X=np.empty((R**3*b_num,3))
        hkl_vectors=np.array([[a,0,0],[0,a,0],[0,0,a]])
        # Find Position of Cells
        hkl_numbers=np.array(list(np.ndindex(R,R,R)))
        hkl_numbers=np.add(hkl_numbers,[1,1,1])
        count=0
        x1_array=np.empty([len(hkl_numbers),3,3])
        for hkl in hkl_numbers:    
            x1_array[count,:,:]=np.multiply(hkl_vectors,hkl)
            count+=1
        x2_array=np.sum(x1_array,axis=1)
        # Save Memory
        del x1_array
        del hkl_numbers
        # Finiding All Atoms Coordinates
        x_array=np.empty([len(x2_array),b_num,3])
        for i in range(b_num):
            x_array[:,i,:]=np.add(x2_array,basis[i])
        X=np.reshape(np.swapaxes(x_array,0,1),(len(x_array)*b_num,3))
        del x_array # Save Memory
        count_id=0
        for id in ID:
                qx=np.sum(np.multiply(X,id),axis=1)
                f=np.add(f,np.cos(qx+phi[count_id]))
                count_id+=1
        f=f*np.emath.sqrt(2/n)
        coor=X[np.where(f<epsilon),:]
        return coor[0]

    def _generate_Waves(self)-> List[List[float]]:
        """
        Generating the Waves of the Nano Porous Material
        
        Returns:
            List of Direction Waves
        """
        H_2=self.h_2
        H=np.emath.sqrt(H_2)
        D=int(np.round(H))
        direc=np.array([[0,0,0]])
        for h in range(-D,D+1):
            for k in range(-D,D+1):
                for l in range(-D,D+1):
                    if np.emath.sqrt(h**2+k**2+l**2)==H:
                        direc=np.append(direc,[[h,k,l]],axis=0)
        direc=np.delete(direc,0,0)
        return direc

    def create_Wave_Directions_File(self,name : str,**kwargs)-> None:
        """
        Generating File of the Direction Waves [name.txt]\n
        Arguments: 
            name: The Name of the Created Lammps File
        Kwargs:
            path : The Path of the New File (defualt: The Path of the Code File)
        Returns:
            None 
        """
        dic_temp={'path' : None}
        dic_temp.update(kwargs)
        if not ((isinstance(dic_temp["path"], str)) | (dic_temp["path"] is None)):
            raise ValueError(f"path must be a string")
        waves_direc=self._generate_Waves()
        if (dic_temp["path"] is None):
            file_path=f"{name}.txt"
        else:
            file_path="/".join([dic_temp["path"],f"{name}.txt"])
        i=1
        with open(file_path, 'w') as f:
            f.write(f"H_2={self.properties['H_2']}\n")
            f.write(f"Number of indepenent waves: {len(waves_direc)}\n")
            f.write("----------------------------------------\n")
            for wave in waves_direc:
                f.write(f"{i}   {wave}\n")
                i+=1
        
    def _generate_List_of_Random_Phi(self,size : int)-> List[float]:
        """
        Generating a list of random phi 
        Arguments:
            size: Number of random phi to make
            path: Where will the File will be Generated
        Returns:
            List of random phi
        """
        phi=np.empty([size,1])
        phi=[rn.uniform(0,2*math.pi) for rand_phi in phi]
        return phi

    def create_Random_Phases_File(self,name : str,path : str="")-> None:
        """ 
        Creating txt File of the Random Phases\n
        Arguments:\n
            name: The Name of the Random Phases File\n
            path (string,optional): The Path of the New File (defualt: The Path of the Code File)
        """
        if path!="":
            file_path=f"{path}/{name}.txt"
        else:
            file_path=f"{name}.txt"
        i=1
        with open(file_path, 'w') as f:
            for rand_phi in self.get_Random_Waves_Phases():
                f.write(f"{i}\t{rand_phi}\n")
                i+=1
                
args = parse_command_line_args()
dataset_path = args['dataset']  
output_dir = args['output_dir']
grid_resolution = args['grid_resolution']
cell_constant = args['cell_constant']
n_cells = args['n_cells']
solid_fraction = args['solid_fraction']    
crystal_system = args['crystal_system']
keep_lmp_files = args['keep_lmp_files']

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
                
df = pd.read_excel(dataset_path)
phases_df = df[df.columns[2:32]]   
                  
num = 0
for i in tqdm(range(0, len(phases_df), 3)):
    def _main() -> None:
        # Generation Parameters - Must Have
        H_2=9 # Wave Number (H^2) -> Wave directions will be made automaticly 
        a_1=cell_constant # Cell Length
        R_1=n_cells # Cell Repitition
        SV_1=solid_fraction # Solid Volume Fraction

        # Optional Parameters
        # Specific Wave Phases: , Defualt - random wave phases    
        phases=phases_df.loc[i].values
        lmp_output = os.path.join(output_dir, 'lmp_output') 
        if not os.path.exists(lmp_output):
            os.makedirs(lmp_output)
        path_lmp = os.path.join(lmp_output, f'lmp_{num}')  

        # Crystal System:
        cs=crystal_system # Choose between 'FCC'/'BCC'/'SC', Default- 'FCC'

        # Generators - Choose & Change to Your Purpose
        npg_g_d=Nano_Porous_Material_Generator(a_1,H_2,R_1,SV_1) # Default generator -> FCC & random wave phases
        npg_g_s=Nano_Porous_Material_Generator(a_1,H_2,R_1,SV_1,random_wave_phases_array=phases,crystal_system=cs) # Specific generator, cystal system is cs & known wave phases
        npg_g_s_2=Nano_Porous_Material_Generator(a_1,H_2,R_1,SV_1,random_wave_phases_array=phases) # Specific generator, FCC & known wave phases
        npg_g_s_3=Nano_Porous_Material_Generator(a_1,H_2,R_1,SV_1,crystal_system=cs) # Specific generator, crystal system is cs & random wave phases

        # Generate- Choose your Generator to generate npg
        my_npg=npg_g_s.generate()

        # Getting Data
            # All Generation Parameters can be extruded by get method from Generator
        h_2=npg_g_d.get_H_2() # Example

            # Getting Random Waves Phases from Generator
        random_phases=npg_g_d.get_Random_Waves_Phases()

            # Getting Atoms' Coordinates of the NPG from the Material
        coor=my_npg.get_Atoms_Coordinates_Matrix()

            # Getting Box Size & Origin
        box_size=my_npg.get_Box_Size()
        origin=my_npg.get_Origin()

            # Creating Files
            ## [Methods are for Linux Systems will not work on Windows] ##
        
        my_npg.create_Lammps_File(f'lmp_{num}', path=lmp_output)                         # Lammps File of the Atoms' Coordianates
        #my_npg.create_Random_Phases_File(name_rpf, path=your_rpf_path)                 File Containing all the Random Wave Phases - uncomment to save
        #my_npg.create_XYZ_File(name_xyz, path=your_xyz_path)                               XYZ File of the Atoms' Coordianates - uncomment to save 
        
        atoms, x_bounds, y_bounds, z_bounds = read_lmp_file(path_lmp + '.txt')
        binary_array = create_binary_grid(atoms, x_bounds, y_bounds, z_bounds, grid_resolution)
        
        npy_output = os.path.join(output_dir, f'grids_{grid_resolution}') 
        if not os.path.exists(npy_output):
            os.makedirs(npy_output)
        
        np.save(os.path.join(npy_output, f'grid_{num}.npy'), binary_array)
        
        if not keep_lmp_files:
            os.remove(path_lmp + '.txt')

    _main()
    num += 1


import torch
import os
import math
import torch.utils.tensorboard

from torch import Tensor
from typing import Tuple, Optional, NamedTuple

from aev import *
from units import *
from nn import *

import torch.utils.data
from collections import defaultdict
import h5py




# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Gaussian(torch.nn.Module):
    """Gaussian activation"""
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(- x * x)


class SpeciesConverter(torch.nn.Module):
    """Converts tensors with species labeled as atomic numbers into tensors
    labeled with internal torchani indices according to a custom ordering
    scheme. It takes a custom species ordering as initialization parameter. If
    the class is initialized with ['H', 'C', 'N', 'O'] for example, it will
    convert a tensor [1, 1, 6, 7, 1, 8] into a tensor [0, 0, 1, 2, 0, 3]

    Arguments:
        species (:class:`collections.abc.Sequence` of :class:`str`):
        sequence of all supported species, in order (it is recommended to order
        according to atomic number).
    """
    conv_tensor: Tensor

    def __init__(self, species):
        super().__init__()
        rev_idx = {s: k for k, s in enumerate(PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())
        self.register_buffer('conv_tensor', torch.full((maxidx + 2,), -1, dtype=torch.long))
        for i, s in enumerate(species):
            self.conv_tensor[rev_idx[s]] = i

    def forward(self, input_: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None):
        """Convert species from periodic table element index to 0, 1, 2, 3, ... indexing"""
        species, coordinates = input_
        converted_species = self.conv_tensor[species]

        # check if unknown species are included
        if converted_species[species.ne(-1)].lt(0).any():
            raise ValueError(f'Unknown species found in {species}')

        return SpeciesCoordinates(converted_species.to(species.device), coordinates)



##################################################################################
"""# Utils"""
##################################################################################


import torch
from torch import Tensor
import torch.utils.data
import math
from collections import defaultdict
from typing import Tuple, NamedTuple, Optional
# from units import sqrt_mhessian2invcm, sqrt_mhessian2milliev, mhessian2fconst
# from nn import SpeciesEnergies


def stack_with_padding(properties, padding):
    output = defaultdict(list)
    for p in properties:
        for k, v in p.items():
            output[k].append(torch.as_tensor(v))
    for k, v in output.items():
        if v[0].dim() == 0:
            output[k] = torch.stack(v)
        else:
            output[k] = torch.nn.utils.rnn.pad_sequence(v, True, padding[k])
    return output


def broadcast_first_dim(properties):
    num_molecule = 1
    for k, v in properties.items():
        shape = list(v.shape)
        n = shape[0]
        if num_molecule != 1:
            assert n == 1 or n == num_molecule, "unable to broadcast"
        else:
            num_molecule = n
    for k, v in properties.items():
        shape = list(v.shape)
        shape[0] = num_molecule
        properties[k] = v.expand(shape)
    return properties


def pad_atomic_properties(properties, padding_values=defaultdict(lambda: 0.0, species=-1)):
    """Put a sequence of atomic properties together into single tensor.

    Inputs are `[{'species': ..., ...}, {'species': ..., ...}, ...]` and the outputs
    are `{'species': padded_tensor, ...}`

    Arguments:
        properties (:class:`collections.abc.Sequence`): sequence of properties.
        padding_values (dict): the value to fill to pad tensors to same size
    """
    vectors = [k for k in properties[0].keys() if properties[0][k].dim() > 1]
    scalars = [k for k in properties[0].keys() if properties[0][k].dim() == 1]
    padded_sizes = {k: max(x[k].shape[1] for x in properties) for k in vectors}
    num_molecules = [x[vectors[0]].shape[0] for x in properties]
    total_num_molecules = sum(num_molecules)
    output = {}
    for k in scalars:
        output[k] = torch.stack([x[k] for x in properties])
    for k in vectors:
        tensor = properties[0][k]
        shape = list(tensor.shape)
        device = tensor.device
        dtype = tensor.dtype
        shape[0] = total_num_molecules
        shape[1] = padded_sizes[k]
        output[k] = torch.full(shape, padding_values[k], device=device, dtype=dtype)
        index0 = 0
        for n, x in zip(num_molecules, properties):
            original_size = x[k].shape[1]
            output[k][index0: index0 + n, 0: original_size, ...] = x[k]
            index0 += n
    return output


def present_species(species):
    """Given a vector of species of atoms, compute the unique species present.

    Arguments:
        species (:class:`torch.Tensor`): 1D vector of shape ``(atoms,)``

    Returns:
        :class:`torch.Tensor`: 1D vector storing present atom types sorted.
    """
    # present_species, _ = species.flatten()._unique(sorted=True)
    present_species = species.flatten().unique(sorted=True)
    if present_species[0].item() == -1:
        present_species = present_species[1:]
    return present_species


def strip_redundant_padding(atomic_properties):
    """Strip trailing padding atoms.

    Arguments:
        atomic_properties (dict): properties to strip

    Returns:
        dict: same set of properties with redundant padding atoms stripped.
    """
    species = atomic_properties['species']
    non_padding = (species >= 0).any(dim=0).nonzero().squeeze()
    for k in atomic_properties:
        atomic_properties[k] = atomic_properties[k].index_select(1, non_padding)
    return atomic_properties


def map2central(cell, coordinates, pbc):
    """Map atoms outside the unit cell into the cell using PBC.

    Arguments:
        cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three
            vectors defining unit cell:

            .. code-block:: python

                tensor([[x1, y1, z1],
                        [x2, y2, z2],
                        [x3, y3, z3]])

        coordinates (:class:`torch.Tensor`): Tensor of shape
            ``(molecules, atoms, 3)``.

        pbc (:class:`torch.Tensor`): boolean vector of size 3 storing
            if pbc is enabled for that direction.

    Returns:
        :class:`torch.Tensor`: coordinates of atoms mapped back to unit cell.
    """
    # Step 1: convert coordinates from standard cartesian coordinate to unit
    # cell coordinates
    inv_cell = torch.inverse(cell)
    coordinates_cell = torch.matmul(coordinates, inv_cell)
    # Step 2: wrap cell coordinates into [0, 1)
    coordinates_cell -= coordinates_cell.floor() * pbc
    # Step 3: convert from cell coordinates back to standard cartesian
    # coordinate
    return torch.matmul(coordinates_cell, cell)



class EnergyShifter(torch.nn.Module):
    """Helper class for adding and subtracting self atomic energies

    This is a subclass of :class:`torch.nn.Module`, so it can be used directly
    in a pipeline as ``[input->AEVComputer->TransformerModel->EnergyShifter->output]``.

    Arguments:
        self_energies (:class:`collections.abc.Sequence`): Sequence of floating
            numbers for the self energy of each atom type. The numbers should
            be in order, i.e. ``self_energies[i]`` should be atom type ``i``.
        fit_intercept (bool): Whether to calculate the intercept during the LSTSQ
            fit. The intercept will also be taken into account to shift energies.
    """

    def __init__(self, self_energies, fit_intercept=False):
        super().__init__()

        self.fit_intercept = fit_intercept
        if self_energies is not None:
            self_energies = torch.tensor(self_energies, dtype=torch.double)

        self.register_buffer('self_energies', self_energies)

    # def __getitem__(self, key):
    #   return self.__dict__(key)

    def sae(self, species):
        """Compute self energies for molecules.

        Padding atoms will be automatically excluded.

        Arguments:
            species (:class:`torch.Tensor`): Long tensor in shape
                ``(conformations, atoms)``.

        Returns:
            :class:`torch.Tensor`: 1D vector in shape ``(conformations,)``
            for molecular self energies.
        """
        intercept = 0.0
        if self.fit_intercept:
            intercept = self.self_energies[-1]

        self_energies = self.self_energies[species].to(species.device)
        # Fix the problem with species in CUDA and self_energies in CPU
        self_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
        return self_energies.sum(dim=1) + intercept

    def forward(self, species_energies: Tuple[Tensor, Tensor],
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        """(species, molecular energies)->(species, molecular energies + sae)
        """
        species, energies = species_energies
        sae = self.sae(species)
        return SpeciesEnergies(species, energies + sae)


class ChemicalSymbolsToInts:
    r"""Helper that can be called to convert chemical symbol string to integers

    On initialization the class should be supplied with a :class:`list` (or in
    general :class:`collections.abc.Sequence`) of :class:`str`. The returned
    instance is a callable object, which can be called with an arbitrary list
    of the supported species that is converted into a tensor of dtype
    :class:`torch.long`. Usage example:

    .. code-block:: python

       from torchani.utils import ChemicalSymbolsToInts


       # We initialize ChemicalSymbolsToInts with the supported species
       species_to_tensor = ChemicalSymbolsToInts(['H', 'C', 'Fe', 'Cl'])

       # We have a species list which we want to convert to an index tensor
       index_tensor = species_to_tensor(['H', 'C', 'H', 'H', 'C', 'Cl', 'Fe'])

       # index_tensor is now [0 1 0 0 1 3 2]


    .. warning::

        If the input is a string python will iterate over
        characters, this means that a string such as 'CHClFe' will be
        intepreted as 'C' 'H' 'C' 'l' 'F' 'e'. It is recommended that you
        input either a :class:`list` or a :class:`numpy.ndarray` ['C', 'H', 'Cl', 'Fe'],
        and not a string. The output of a call does NOT correspond to a
        tensor of atomic numbers.

    Arguments:
        all_species (:class:`collections.abc.Sequence` of :class:`str`):
        sequence of all supported species, in order (it is recommended to order
        according to atomic number).
    """

    def __init__(self, all_species):
        self.rev_species = {s: i for i, s in enumerate(all_species)}

    def __call__(self, species):
        r"""Convert species from sequence of strings to 1D tensor"""
        rev = [self.rev_species[s] for s in species]
        return torch.tensor(rev, dtype=torch.long)

    def __len__(self):
        return len(self.rev_species)


def _get_derivatives_not_none(x: Tensor, y: Tensor, retain_graph: Optional[bool] = None, create_graph: bool = False) -> Tensor:
    ret = torch.autograd.grad([y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph)[0]
    assert ret is not None
    return ret


def hessian(coordinates: Tensor, energies: Optional[Tensor] = None, forces: Optional[Tensor] = None) -> Tensor:
    """Compute analytical hessian from the energy graph or force graph.

    Arguments:
        coordinates (:class:`torch.Tensor`): Tensor of shape `(molecules, atoms, 3)`
        energies (:class:`torch.Tensor`): Tensor of shape `(molecules,)`, if specified,
            then `forces` must be `None`. This energies must be computed from
            `coordinates` in a graph.
        forces (:class:`torch.Tensor`): Tensor of shape `(molecules, atoms, 3)`, if specified,
            then `energies` must be `None`. This forces must be computed from
            `coordinates` in a graph.

    Returns:
        :class:`torch.Tensor`: Tensor of shape `(molecules, 3A, 3A)` where A is the number of
        atoms in each molecule
    """
    if energies is None and forces is None:
        raise ValueError('Energies or forces must be specified')
    if energies is not None and forces is not None:
        raise ValueError('Energies or forces can not be specified at the same time')
    if forces is None:
        assert energies is not None
        forces = -_get_derivatives_not_none(coordinates, energies, create_graph=True)
    flattened_force = forces.flatten(start_dim=1)
    force_components = flattened_force.unbind(dim=1)
    return -torch.stack([
        _get_derivatives_not_none(coordinates, f, retain_graph=True).flatten(start_dim=1)
        for f in force_components
    ], dim=1)


class FreqsModes(NamedTuple):
    freqs: Tensor
    modes: Tensor


class VibAnalysis(NamedTuple):
    freqs: Tensor
    modes: Tensor
    fconstants: Tensor
    rmasses: Tensor


def vibrational_analysis(masses, hessian, mode_type='MDU', unit='cm^-1'):
    """Computing the vibrational wavenumbers from hessian.

    Note that normal modes in many popular software packages such as
    Gaussian and ORCA are output as mass deweighted normalized (MDN).
    Normal modes in ASE are output as mass deweighted unnormalized (MDU).
    Some packages such as Psi4 let ychoose different normalizations.
    Force constants and reduced masses are calculated as in Gaussian.

    mode_type should be one of:
    - MWN (mass weighted normalized)
    - MDU (mass deweighted unnormalized)
    - MDN (mass deweighted normalized)

    MDU modes are not orthogonal, and not normalized,
    MDN modes are not orthogonal, and normalized.
    MWN modes are orthonormal, but they correspond
    to mass weighted cartesian coordinates (x' = sqrt(m)x).
    """
    if unit == 'meV':
        unit_converter = sqrt_mhessian2milliev
    elif unit == 'cm^-1':
        unit_converter = sqrt_mhessian2invcm
    else:
        raise ValueError('Only meV and cm^-1 are supported right now')

    assert hessian.shape[0] == 1, 'Currently only supporting computing one molecule a time'
    # Solving the eigenvalue problem: Hq = w^2 * T q
    # where H is the Hessian matrix, q is the normal coordinates,
    # T = diag(m1, m1, m1, m2, m2, m2, ....) is the mass
    # We solve this eigenvalue problem through Lowdin diagnolization:
    # Hq = w^2 * Tq ==> Hq = w^2 * T^(1/2) T^(1/2) q
    # Letting q' = T^(1/2) q, we then have
    # T^(-1/2) H T^(-1/2) q' = w^2 * q'
    inv_sqrt_mass = (1 / masses.sqrt()).repeat_interleave(3, dim=1)  # shape (molecule, 3 * atoms)
    mass_scaled_hessian = hessian * inv_sqrt_mass.unsqueeze(1) * inv_sqrt_mass.unsqueeze(2)
    if mass_scaled_hessian.shape[0] != 1:
        raise ValueError('The input should contain only one molecule')
    mass_scaled_hessian = mass_scaled_hessian.squeeze(0)
    eigenvalues, eigenvectors = torch.symeig(mass_scaled_hessian, eigenvectors=True)
    angular_frequencies = eigenvalues.sqrt()
    frequencies = angular_frequencies / (2 * math.pi)
    # converting from sqrt(hartree / (amu * angstrom^2)) to cm^-1 or meV
    wavenumbers = unit_converter(frequencies)

    # Note that the normal modes are the COLUMNS of the eigenvectors matrix
    mw_normalized = eigenvectors.t()
    md_unnormalized = mw_normalized * inv_sqrt_mass
    norm_factors = 1 / torch.linalg.norm(md_unnormalized, dim=1)  # units are sqrt(AMU)
    md_normalized = md_unnormalized * norm_factors.unsqueeze(1)

    rmasses = norm_factors**2  # units are AMU
    # The conversion factor for Ha/(AMU*A^2) to mDyne/(A*AMU) is about 4.3597482
    fconstants = mhessian2fconst(eigenvalues) * rmasses  # units are mDyne/A

    if mode_type == 'MDN':
        modes = (md_normalized).reshape(frequencies.numel(), -1, 3)
    elif mode_type == 'MDU':
        modes = (md_unnormalized).reshape(frequencies.numel(), -1, 3)
    elif mode_type == 'MWN':
        modes = (mw_normalized).reshape(frequencies.numel(), -1, 3)

    return VibAnalysis(wavenumbers, modes, fconstants, rmasses)


def get_atomic_masses(species):
    r"""Convert a tensor of atomic numbers ("periodic table indices") into a tensor of atomic masses

    Atomic masses supported are the first 119 elements, and are taken from:

    Atomic weights of the elements 2013 (IUPAC Technical Report). Meija, J.,
    Coplen, T., Berglund, M., et al. (2016). Pure and Applied Chemistry, 88(3), pp.
    265-291. Retrieved 30 Nov. 2016, from doi:10.1515/pac-2015-0305

    They are all consistent with those used in ASE

    Arguments:
        species (:class:`torch.Tensor`): tensor with atomic numbers

    Returns:
        :class:`torch.Tensor`: Tensor of dtype :class:`torch.double`, with
        atomic masses, with the same shape as the input.
    """
    # Note that there should not be any atoms with index zero, because that is
    # not an element
    assert len((species == 0).nonzero()) == 0
    default_atomic_masses = torch.tensor(
            [0.        ,   1.008     ,   4.002602  ,   6.94      , # noqa
             9.0121831 ,  10.81      ,  12.011     ,  14.007     , # noqa
            15.999     ,  18.99840316,  20.1797    ,  22.98976928, # noqa
            24.305     ,  26.9815385 ,  28.085     ,  30.973762  , # noqa
            32.06      ,  35.45      ,  39.948     ,  39.0983    , # noqa
            40.078     ,  44.955908  ,  47.867     ,  50.9415    , # noqa
            51.9961    ,  54.938044  ,  55.845     ,  58.933194  , # noqa
            58.6934    ,  63.546     ,  65.38      ,  69.723     , # noqa
            72.63      ,  74.921595  ,  78.971     ,  79.904     , # noqa
            83.798     ,  85.4678    ,  87.62      ,  88.90584   , # noqa
            91.224     ,  92.90637   ,  95.95      ,  97.90721   , # noqa
           101.07      , 102.9055    , 106.42      , 107.8682    , # noqa
           112.414     , 114.818     , 118.71      , 121.76      , # noqa
           127.6       , 126.90447   , 131.293     , 132.90545196, # noqa
           137.327     , 138.90547   , 140.116     , 140.90766   , # noqa
           144.242     , 144.91276   , 150.36      , 151.964     , # noqa
           157.25      , 158.92535   , 162.5       , 164.93033   , # noqa
           167.259     , 168.93422   , 173.054     , 174.9668    , # noqa
           178.49      , 180.94788   , 183.84      , 186.207     , # noqa
           190.23      , 192.217     , 195.084     , 196.966569  , # noqa
           200.592     , 204.38      , 207.2       , 208.9804    , # noqa
           208.98243   , 209.98715   , 222.01758   , 223.01974   , # noqa
           226.02541   , 227.02775   , 232.0377    , 231.03588   , # noqa
           238.02891   , 237.04817   , 244.06421   , 243.06138   , # noqa
           247.07035   , 247.07031   , 251.07959   , 252.083     , # noqa
           257.09511   , 258.09843   , 259.101     , 262.11      , # noqa
           267.122     , 268.126     , 271.134     , 270.133     , # noqa
           269.1338    , 278.156     , 281.165     , 281.166     , # noqa
           285.177     , 286.182     , 289.19      , 289.194     , # noqa
           293.204     , 293.208     , 294.214], # noqa
        dtype=torch.double, device=species.device) # noqa
    masses = default_atomic_masses[species]
    return masses


# This constant, when indexed with the corresponding atomic number, gives the
# element associated with it. Note that there is no element with atomic number
# 0, so 'Dummy' returned in this case.
PERIODIC_TABLE = ['Dummy'] + """
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()


__all__ = ['pad_atomic_properties', 'present_species', 'hessian',
           'vibrational_analysis', 'strip_redundant_padding',
           'ChemicalSymbolsToInts', 'get_atomic_masses']

"""# Data Loader"""

##################################################################################
"""# Data Loader File"""
##################################################################################


# Written by Roman Zubatyuk and Justin S. Smith
import h5py
import numpy as np
import os


class datapacker:
    def __init__(self, store_file, mode='w-', complib='gzip', complevel=6):
        """Wrapper to store arrays within HFD5 file"""
        self.store = h5py.File(store_file, mode=mode)
        self.clib = complib
        self.clev = complevel

    def store_data(self, store_loc, **kwargs):
        """Put arrays to store"""
        g = self.store.create_group(store_loc)
        for k, v, in kwargs.items():
            if isinstance(v, list):
                if len(v) != 0:
                    if isinstance(v[0], np.str_) or isinstance(v[0], str):
                        v = [a.encode('utf8') for a in v]

            g.create_dataset(k, data=v, compression=self.clib,
                             compression_opts=self.clev)

    def cleanup(self):
        """Wrapper to close HDF5 file"""
        self.store.close()


class anidataloader:

    def __init__(self, store_file):
        if not os.path.exists(store_file):
            exit('Error: file not found - ' + store_file)
        self.store = h5py.File(store_file, 'r')

    def h5py_dataset_iterator(self, g, prefix=''):
        """Group recursive iterator

        Iterate through all groups in all branches and return datasets in dicts)
        """
        for key in g.keys():
            item = g[key]
            path = '{}/{}'.format(prefix, key)
            keys = [i for i in item.keys()]
            if isinstance(item[keys[0]], h5py.Dataset):  # test for dataset
                data = {'path': path}
                for k in keys:
                    if not isinstance(item[k], h5py.Group):
                        dataset = np.array(item[k][()])

                        if isinstance(dataset, np.ndarray):
                            if dataset.size != 0:
                                if isinstance(dataset[0], np.bytes_):
                                    dataset = [a.decode('ascii')
                                               for a in dataset]
                        data.update({k: dataset})
                yield data
            else:  # test for group (go down)
                yield from self.h5py_dataset_iterator(item, path)

    def __iter__(self):
        """Default class iterator (iterate through all data)"""
        for data in self.h5py_dataset_iterator(self.store):
            yield data

    def get_group_list(self):
        """Returns a list of all groups in the file"""
        return [g for g in self.store.values()]

    def iter_group(self, g):
        """Allows interation through the data in a given group"""
        for data in self.h5py_dataset_iterator(g):
            yield data

    def get_data(self, path, prefix=''):
        """Returns the requested dataset"""
        item = self.store[path]
        path = '{}/{}'.format(prefix, path)
        keys = [i for i in item.keys()]
        data = {'path': path}
        for k in keys:
            if not isinstance(item[k], h5py.Group):
                dataset = np.array(item[k][()])

                if isinstance(dataset, np.ndarray):
                    if dataset.size != 0:
                        if isinstance(dataset[0], np.bytes_):
                            dataset = [a.decode('ascii') for a in dataset]

                data.update({k: dataset})
        return data

    def group_size(self):
        """Returns the number of groups"""
        return len(self.get_group_list())

    def size(self):
        count = 0
        for g in self.store.values():
            count = count + len(g.items())
        return count

    def cleanup(self):
        """Close the HDF5 file"""
        self.store.close()

# -*- coding: utf-8 -*-
"""Tools for loading, shuffling, and batching ANI datasets

The `torchani.data.load(path)` creates an iterable of raw data,
where species are strings, and coordinates are numpy ndarrays.

You can transform this iterable by using transformations.
To do a transformation, call `it.transformation_name()`. This
will return an iterable that may be cached depending on the specific
transformation.

Available transformations are listed below:

- `species_to_indices` accepts two different kinds of arguments. It converts
    species from elements (e. g. "H", "C", "Cl", etc) into internal torchani
    indices (as returned by :class:`torchani.utils.ChemicalSymbolsToInts` or
    the ``species_to_tensor`` method of a :class:`torchani.models.BuiltinModel`
    and :class:`torchani.neurochem.Constants`), if its argument is an iterable
    of species. By default species_to_indices behaves this way, with an
    argument of ``('H', 'C', 'N', 'O', 'F', 'S', 'Cl')``  However, if its
    argument is the string "periodic_table", then elements are converted into
    atomic numbers ("periodic table indices") instead. This last option is
    meant to be used when training networks that already perform a forward pass
    of :class:`torchani.nn.SpeciesConverter` on their inputs in order to
    convert elements to internal indices, before processing the coordinates.

- `subtract_self_energies` subtracts self energies from all molecules of the
    dataset. It accepts two different kinds of arguments: You can pass a dict
    of self energies, in which case self energies are directly subtracted
    according to the key-value pairs, or a
    :class:`torchani.utils.EnergyShifter`, in which case the self energies are
    calculated by linear regression and stored inside the class in the order
    specified by species_order. By default the function orders by atomic
    number if no extra argument is provided, but a specific order may be requested.

- `remove_outliers` removes some outlier energies from the dataset if present.

- `shuffle` shuffles the provided dataset. Note that if the dataset is
    not cached (i.e. it lives in the disk and not in memory) then this method
    will cache it before shuffling. This may take time and memory depending on
    the dataset size. This method may be used before splitting into validation/training
    shuffle all molecules in the dataset, and ensure a uniform sampling from
    the initial dataset, and it can also be used during training on a cached
    dataset of batches to shuffle the batches.

- `cache` cache the result of previous transformations.
    If the input is already cached this does nothing.

- `collate` creates batches and pads the atoms of all molecules in each batch
    with dummy atoms, then converts each batch to tensor. `collate` uses a
    default padding dictionary:
    ``{'species': -1, 'coordinates': 0.0, 'forces': 0.0, 'energies': 0.0}`` for
    padding, but a custom padding dictionary can be passed as an optional
    parameter, which overrides this default padding. Note that this function
    returns a generator, it doesn't cache the result in memory.

- `pin_memory` copies the tensor to pinned (page-locked) memory so that later transfer
    to cuda devices can be done faster.

you can also use `split` to split the iterable to pieces. use `split` as:

.. code-block:: python

    it.split(ratio1, ratio2, None)

where None in the end indicate that we want to use all of the rest.

Note that orderings used in :class:`torchani.utils.ChemicalSymbolsToInts` and
:class:`torchani.nn.SpeciesConverter` should be consistent with orderings used
in `species_to_indices` and `subtract_self_energies`. To prevent confusion it
is recommended that arguments to intialize converters and arguments to these
functions all order elements *by their atomic number* (e. g. if you are working
with hydrogen, nitrogen and bromine always use ['H', 'N', 'Br'] and never ['N',
'H', 'Br'] or other variations). It is possible to specify a different custom
ordering, mainly due to backwards compatibility and to fully custom atom types,
but doing so is NOT recommended, since it is very error prone.


Example:

.. code-block:: python

    energy_shifter = torchani.utils.EnergyShifter(None)
    training, validation = torchani.data.load(dspath).subtract_self_energies(energy_shifter).species_to_indices().shuffle().split(int(0.8 * size), None)
    training = training.collate(batch_size).cache()
    validation = validation.collate(batch_size).cache()

If the above approach takes too much memory for you, you can then use dataloader
with multiprocessing to achieve comparable performance with less memory usage:

.. code-block:: python

    training, validation = torchani.data.load(dspath).subtract_self_energies(energy_shifter).species_to_indices().shuffle().split(0.8, None)
    training = torch.utils.data.DataLoader(list(training), batch_size=batch_size, collate_fn=torchani.data.collate_fn, num_workers=64)
    validation = torch.utils.data.DataLoader(list(validation), batch_size=batch_size, collate_fn=torchani.data.collate_fn, num_workers=64)
"""

from os.path import join, isfile, isdir
import os
# from ._pyanitools import anidataloader
# from .. import utils
import importlib
import functools
import math
import random
from collections import Counter
import numpy
import gc

PKBAR_INSTALLED = importlib.util.find_spec('pkbar') is not None  # type: ignore
if PKBAR_INSTALLED:
    import pkbar

verbose = True

PROPERTIES = ('energies',)

PADDING = {
    'species': -1,
    'coordinates': 0.0,
    'forces': 0.0,
    'energies': 0.0
}


def collate_fn(samples, padding=None):
    if padding is None:
        padding = PADDING

    return stack_with_padding(samples, padding)


class IterableAdapter:
    """https://stackoverflow.com/a/39564774"""
    def __init__(self, iterable_factory, length=None):
        self.iterable_factory = iterable_factory
        self.length = length

    def __iter__(self):
        return iter(self.iterable_factory())


class IterableAdapterWithLength(IterableAdapter):

    def __init__(self, iterable_factory, length):
        super().__init__(iterable_factory)
        self.length = length

    def __len__(self):
        return self.length


class Transformations:
    """Convert one reenterable iterable to another reenterable iterable"""

    @staticmethod
    def species_to_indices(reenterable_iterable, species_order=('H', 'C', 'N', 'O', 'S', 'Cl')):
        if species_order == 'periodic_table':
            species_order = PERIODIC_TABLE
        idx = {k: i for i, k in enumerate(species_order)}

        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                d['species'] = numpy.array([idx[s] for s in d['species']], dtype='i8')
                yield d
        try:
            return IterableAdapterWithLength(reenterable_iterable_factory, len(reenterable_iterable))
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)

    @staticmethod
    def subtract_self_energies(reenterable_iterable, self_energies=None, species_order=None):
        intercept = 0.0
        shape_inference = False
        if isinstance(self_energies, EnergyShifter):
            shape_inference = True
            shifter = self_energies
            self_energies = {}
            counts = {}
            Y = []
            for n, d in enumerate(reenterable_iterable):
                species = d['species']
                count = Counter()
                for s in species:
                    count[s] += 1
                for s, c in count.items():
                    if s not in counts:
                        counts[s] = [0] * n
                    counts[s].append(c)
                for s in counts:
                    if len(counts[s]) != n + 1:
                        counts[s].append(0)
                Y.append(d['energies'])

            # sort based on the order in periodic table by default
            if species_order is None:
                species_order = PERIODIC_TABLE

            species = sorted(list(counts.keys()), key=lambda x: species_order.index(x))

            X = [counts[s] for s in species]
            if shifter.fit_intercept:
                X.append([1] * n)
            X = numpy.array(X).transpose()
            Y = numpy.array(Y)
            if Y.shape[0] == 0:
                raise RuntimeError("subtract_self_energies could not find any energies in the provided dataset.\n"
                                   "Please make sure the path provided to data.load() points to a dataset has energies and is not empty or corrupted.")
            sae, _, _, _ = numpy.linalg.lstsq(X, Y, rcond=None)
            sae_ = sae
            if shifter.fit_intercept:
                intercept = sae[-1]
                sae_ = sae[:-1]
            for s, e in zip(species, sae_):
                self_energies[s] = e
            shifter.__init__(sae, shifter.fit_intercept)
        gc.collect()

        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                e = intercept
                for s in d['species']:
                    e += self_energies[s]
                d['energies'] -= e
                yield d
        if shape_inference:
            return IterableAdapterWithLength(reenterable_iterable_factory, n)
        return IterableAdapter(reenterable_iterable_factory)

    @staticmethod
    def remove_outliers(reenterable_iterable, threshold1=15.0, threshold2=8.0):
        assert 'subtract_self_energies', "Transformation remove_outliers can only run after subtract_self_energies"

        # pass 1: remove everything that has per-atom energy > threshold1
        def scaled_energy(x):
            num_atoms = len(x['species'])
            return abs(x['energies']) / math.sqrt(num_atoms)
        filtered = IterableAdapter(lambda: (x for x in reenterable_iterable if scaled_energy(x) < threshold1))

        # pass 2: compute those that are outside the mean by threshold2 * std
        n = 0
        mean = 0
        std = 0
        for m in filtered:
            n += 1
            mean += m['energies']
            std += m['energies'] ** 2
        mean /= n
        std = math.sqrt(std / n - mean ** 2)

        return IterableAdapter(lambda: filter(lambda x: abs(x['energies'] - mean) < threshold2 * std, filtered))

    @staticmethod
    def shuffle(reenterable_iterable):
        if isinstance(reenterable_iterable, list):
            list_ = reenterable_iterable
        else:
            list_ = list(reenterable_iterable)
            del reenterable_iterable
            gc.collect()
        random.shuffle(list_)
        return list_

    @staticmethod
    def cache(reenterable_iterable):
        if isinstance(reenterable_iterable, list):
            return reenterable_iterable
        ret = list(reenterable_iterable)
        del reenterable_iterable
        gc.collect()
        return ret

    @staticmethod
    def collate(reenterable_iterable, batch_size, padding=None):
        def reenterable_iterable_factory(padding=None):
            batch = []
            i = 0
            for d in reenterable_iterable:
                batch.append(d)
                i += 1
                if i == batch_size:
                    i = 0
                    yield collate_fn(batch, padding)
                    batch = []
            if len(batch) > 0:
                yield collate_fn(batch, padding)

        reenterable_iterable_factory = functools.partial(reenterable_iterable_factory,
                                                         padding)
        try:
            length = (len(reenterable_iterable) + batch_size - 1) // batch_size
            return IterableAdapterWithLength(reenterable_iterable_factory, length)
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)

    @staticmethod
    def pin_memory(reenterable_iterable):
        def reenterable_iterable_factory():
            for d in reenterable_iterable:
                yield {k: d[k].pin_memory() for k in d}
        try:
            return IterableAdapterWithLength(reenterable_iterable_factory, len(reenterable_iterable))
        except TypeError:
            return IterableAdapter(reenterable_iterable_factory)


class TransformableIterable:
    def __init__(self, wrapped_iterable, transformations=()):
        self.wrapped_iterable = wrapped_iterable
        self.transformations = transformations

    def __iter__(self):
        return iter(self.wrapped_iterable)

    def __getattr__(self, name):
        transformation = getattr(Transformations, name)

        @functools.wraps(transformation)
        def f(*args, **kwargs):
            return TransformableIterable(
                transformation(self.wrapped_iterable, *args, **kwargs),
                self.transformations + (name,))

        return f

    def split(self, *nums):
        length = len(self)
        iters = []
        self_iter = iter(self)
        for n in nums:
            list_ = []
            if n is not None:
                for _ in range(int(n * length)):
                    list_.append(next(self_iter))
            else:
                for i in self_iter:
                    list_.append(i)
            iters.append(TransformableIterable(list_, self.transformations + ('split',)))
        del self_iter
        gc.collect()
        return iters

    def __len__(self):
        return len(self.wrapped_iterable)


def load(path, additional_properties=()):
    properties = PROPERTIES + additional_properties

    def h5_files(path):
        """yield file name of all h5 files in a path"""
        if isdir(path):
            for f in os.listdir(path):
                f = join(path, f)
                yield from h5_files(f)
        elif isfile(path) and (path.endswith('.h5') or path.endswith('.hdf5')):
            yield path

    def molecules():
        for f in h5_files(path):
            anidata = anidataloader(f)
            anidata_size = anidata.group_size()
            use_pbar = PKBAR_INSTALLED and verbose
            if use_pbar:
                pbar = pkbar.Pbar('=> loading {}, total molecules: {}'.format(f, anidata_size), anidata_size)
            for i, m in enumerate(anidata):
                yield m
                if use_pbar:
                    pbar.update(i)

    def conformations():
        for m in molecules():
            species = m['species']
            coordinates = m['coordinates']
            for i in range(coordinates.shape[0]):
                ret = {'species': species, 'coordinates': coordinates[i]}
                for k in properties:
                    if k in m:
                        ret[k] = m[k][i]
                yield ret

    return TransformableIterable(IterableAdapter(lambda: conformations()))


__all__ = ['load', 'collate_fn']
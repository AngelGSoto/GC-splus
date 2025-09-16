import ConfigParser
import ast
import os
# from magal.library import LibraryModel

__author__ = 'william'

class MagalConfig(ConfigParser.RawConfigParser):
    """
    This class abstracts the configuration files and checks for all the MagAl scripts.
    """

    def __init__(self, config_file, config_type):
        """
        Init a MagAl config instance.
        Parameters
        ----------
        config_file : string
            Path to the configuration file
        config_type : string
            Means which script is calling this object.
        See Also
        --------
        ConfigParser.RawConfigParser
        """
        ConfigParser.RawConfigParser.__init__(self)

        ret = 0
        if not os.path.exists(config_file):
            raise IOError('File %s does not exists.' % config_file)
        self.read(config_file)
        if config_type == 'fit':
            self.ret = self._load_fit_vars()
        elif config_type == 'library':
            self._load_library_vars()
        elif config_type == 'mkfilterset':
            self._load_mkfilterset_vars()
        else:
            print 'Error! Invalid config_type %s.' % config_type
            self.ret = 1

    def _check_output_file(self, filename, allow_overwrite=False):
        '''
        Checks is outputfile exists or not. If allow_overwirte = True, then it deletes the file.
        :param filename:
        :param allow_overwrite:
        :return:
        '''
        if os.path.exists(filename):
            if allow_overwrite:
                try:
                    os.unlink(self.output_file)
                except:
                    print u'Cannot DELETE file {0:s}.'.format(filename)
                    return 1
            else:
                print(u'File {0:s} already exists.'.format(filename))
                return 1
        else:
            self.ret = 0

    def _load_fit_vars(self):
        """
        Reads configuration file to magal_fit.
        """
        self.input_file = os.path.expandvars(self.get('FitGeneral', 'input_file'))  # Input File
        self.library_file = os.path.expandvars(self.get('FitGeneral', 'library_file'))  # Template ibrary File
        self.output_file = os.path.expandvars(self.get('FitGeneral', 'output_file'))  # Output File
        self.filter_sys = self.get('FitGeneral', 'filter_sys')  # Filtersystem (e.g. SDSS)
        try:
            self.ccd = self.get('FitGeneral', 'ccd')  # CCD number
        except ConfigParser.NoOptionError:
            print 'Magal fit over all CCDs of inputfile!'
            self.ccd = None

        #Multiprocessing: Default is True.
        try:
            self.allow_multiprocessing = self.getboolean('FitGeneral', 'allow_multiprocessing')
        except ConfigParser.NoOptionError:
            self.allow_multiprocessing = True

        #If allow_overwrite is defined, we can overwrite the outputfile. Default: False.
        try:
            allow_overwrite = self.getboolean('FitGeneral', 'allow_overwrite')
        except ConfigParser.NoOptionError:
            allow_overwrite = False
        #Check if outputfile exists and act as defined by config.
        ret = self._check_output_file(self.output_file, allow_overwrite)
        if ret != 0:
            self.ret = ret

        # Look if there is a filter list to include.
        try:
            self.filters_include = ast.literal_eval(self.get('FitGeneral', 'filters_include'))
        except ConfigParser.NoOptionError:
            print('Fitting for all the filters.')
            self.filters_include = None
        try:
            filters_exclude = ast.literal_eval(self.get('FitGeneral', 'filters_exclude'))
        except ConfigParser.NoOptionError:
            print('Fitting for all the filters.')
            self.filters_exclude = None

        if self.filters_include is not None and self.filters_exclude is not None:
            print 'Error. You cannot have filters_include and filters_exclude Config Options at the same time.'
            self.ret = 1

        try:
            self.zp_error = ast.literal_eval(self.get('FitGeneral', 'zp_error'))
        except ConfigParser.NoOptionError:
            self.zp_error = None
            print('zp_error is not defined on the .ini file!')

        # Look if there is a maximum number of objects to run. This is useful for testing.
        try:
            self.Nobj_max = self.getint('FitGeneral', 'Nobj_max')
        except ConfigParser.NoOptionError:
            self.Nobj_max = 0
            print('Running magal fit to ALL objects.')

        # Cooking factor. It is multiplied by the chi2 in order to increase the effectiveness of each template
        # on the likelihood.
        try:
            self.cooking_factor = self.getfloat('FitGeneral', 'cooking_factor')
            print('Found a cooking_factor option. \chi^2 will be multiplied by %3.2f' % self.cooking_factor)
        except ConfigParser.NoOptionError:
            self.cooking_factor = None

        # Check if the user request a simulation. For a simulation, the inputfile will be a Library template file.
        try:
            self.is_simulation = self.getboolean('FitSimulation', 'is_simulation')  # Is this a simulation?
            if self.is_simulation:
                try:
                    self.obj_z = self.getfloat('FitSimulation', 'obj_z')
                    print('Using z = %s from inputfile on this simulation.' % self.obj_z)
                except ConfigParser.NoOptionError:
                    self.obj_z = False
                    print('Using z from inputfile tables.')
                try:
                    self.mass_field = self.get('FitSimulation', 'mass_field')
                except ConfigParser.NoOptionError:
                    print('When running a simulation, you must specify a table field for the mass.')
                    self.ret = 1
            else:
                self.mass_field = None
        except ConfigParser.NoSectionError:
            self.is_simulation = False  # If there is no [FitSimulation] on self, just ignore this.

        try:
            self.Nz = self.getboolean('FitGeneral', 'Nz')  # If True, will fit at z = const.
            # Redshift will be get from inputfile property z.
        except ConfigParser.NoOptionError:
            if self.is_simulation:
                self.Nz = False
            else:
                print('Evaluating for all redshift space!')

        self.ret = 0



    def _get_inputfiles(self, section, files):
        """
        Parameters
        ----------
        section : string
            config section to get the filenames.
        files : list
            Filenames list.
        Returns
        -------
        input_files : tuple
            All the asked inputfiles.
        Raises
        ------
        """
        fnames = tuple()
        for f_ in files:
            try:
                fnames += (ast.literal_eval(os.path.expandvars(self.get(section, f_))),)
            except ValueError:
                fnames += (os.path.expandvars(self.get(section, f_)),)
        return fnames


    def _load_library_vars(self):
        """
        Loads magal_library related configfile parameters.
        """
        libfile = ''
        library_type = os.path.expandvars(self.get('LibraryGeneral', 'library_type'))
        filter_file = os.path.expandvars(self.get('LibraryGeneral', 'filter_file'))

        filter_id = self.get('LibraryGeneral', 'filterset')  #TODO: Check if it is valid.
        z_from = self.getfloat('LibraryGeneral', 'z_from')
        z_to = self.getfloat('LibraryGeneral', 'z_to')
        z_step = self.getfloat('LibraryGeneral', 'z_step')
        try:
            cosmology = ast.literal_eval(self.get('LibraryGeneral', 'cosmology'))
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=cosmology['H0'], Om0=cosmology['Om0'])
        except ConfigParser.NoOptionError:
            print 'Adopting the WMAP9 comology:'
            from astropy.cosmology import WMAP9 as cosmo
            print cosmo.__doc__

        try:
            allow_overwrite = self.getboolean('LibraryGeneral', 'allow_overwrite')
        except ConfigParser.NoOptionError:
            allow_overwrite = False
        if os.path.exists(libfile):
            if allow_overwrite:
                try:
                    os.unlink(libfile)
                except:
                    print u'Cannot DELETE file {0:s}.'.format(libfile)
                    # return 1
            else:
                print u'File {0:s} already exists.'.format(libfile)
                # return 1


        # 1 - Two-exponential library parameters:
        if library_type == 'two_exp':
            inp_file = os.path.expandvars(self.get('LibraryParameters', 'input_file'))
            basedir = os.path.expandvars(self.get('LibraryParameters', 'bases_dir'))
            basefile = os.path.expandvars(self.get('LibraryParameters', 'base_file'))
            fraction_type = self.get('LibraryParameters', 'fraction_type')  # light or mass
            try:
                lambda_norm = self.getfloat('LibraryParameters', 'lambda_norm')  # mass normalization wl
            except ConfigParser.NoOptionError:
                lambda_norm = 4020
            LibModel = LibraryModel(library_type, inp_file, basedir, basefile, fraction_type, lambda_norm)  # Init
        # 2 - STARLIGHT library parameters:
        elif library_type == 'starlight_sdss':
            type = 'starlight_sdss'
            inp_file = os.path.expandvars(self.get('LibraryParameters', 'input_file'))
            tables_dir = os.path.expandvars(self.get('LibraryParameters', 'tables_dir'))
            input_dir = os.path.expandvars(self.get('LibraryParameters', 'input_dir'))
            output_dir = os.path.expandvars(self.get('LibraryParameters', 'output_dir'))
            LibModel = LibraryModel(type, inp_file, tables_dir, input_dir, output_dir)

        #### -- ####


    def _load_mkfilterset_vars(self):
        '''
        Loads magal_mkfilter related vars
        '''

        # 1 - Get the filenames
        self.filterset_file, self.filter_filenames = self._get_inputfiles('FilterLibrary',
                                                                          ('filterset_file', 'filter_filenames'))
        # 2 - Check if filters exists
        for ccd in self.filter_filenames.keys():
            for filter_file in self.filter_filenames[ccd]:
                if not os.path.isfile(filter_file):
                    print 'File %s does not exists' % filter_file
                    return False
        if os.path.exists(self.filterset_file):
            print 'File %s does exists' % self.filterset_file
            return False

        # 3 - Read the other options
        self.filterset_name = self.get('FilterLibrary', 'filterset_name')
        self.filter_names = ast.literal_eval(self.get('FilterLibrary', 'filter_names'))

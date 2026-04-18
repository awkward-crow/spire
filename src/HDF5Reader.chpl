/*
  HDF5Reader.chpl
  ---------------
  Reads a GBMData record from an HDF5 file.

  Expected HDF5 layout:
    /X   float32[nSamples, nFeatures]   feature matrix, row-major
    /y   float32[nSamples]              labels (0.0 or 1.0)

  Each locale opens the file independently and reads its own slice via
  HDF5 hyperslab selection — no locale-0 bottleneck, no MPI required.
  This is the same approach as HDF5.IOusingMPI.hdf5ReadDistributedArray
  but uses independent (non-collective) I/O so the parallel HDF5 library
  is not needed.

  Compile with: -lhdf5
*/

module HDF5Reader {

  use HDF5;
  use DataLayout;

  proc readHDF5(filename: string): GBMData {
    use C_HDF5;
    use C_HDF5.HDF5_WAR;

    // ---- Read dataset dimensions on locale 0 ----------------------------
    var file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if file_id < 0 then
      halt("HDF5Reader: cannot open " + filename);

    var xRank: c_int;
    H5LTget_dataset_ndims(file_id, "X".c_str(), xRank);
    var xDims: [1..xRank] hsize_t;
    H5LTget_dataset_info_WAR(file_id, "X".c_str(),
                              c_ptrTo(xDims[1]), nil, nil);
    H5Fclose(file_id);

    const nSamples  = xDims[1]: int;
    const nFeatures = xDims[2]: int;

    var data = new GBMData(numSamples=nSamples, numFeatures=nFeatures);

    // ---- Per-locale hyperslab reads -------------------------------------
    // Each locale opens the file and reads its local subdomain directly.
    // Independent (non-collective) I/O — no MPI required.
    coforall loc in Locales with (ref data) {
      on loc {
        var localFilename = filename;   // make filename local to this locale
        var fid = H5Fopen(localFilename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

        // ---- X ----------------------------------------------------------
        {
          var dset      = H5Dopen(fid, "X".c_str(), H5P_DEFAULT);
          var filespace = H5Dget_space(dset);

          for dom in data.X.localSubdomains() {
            // dom is e.g. {rowLow..rowHigh, colLow..colHigh}
            var offset: [0..1] hsize_t;
            var count:  [0..1] hsize_t;
            for param i in 0..1 {
              offset[i] = dom.dim(i).low:  hsize_t;
              count[i]  = dom.dim(i).size: hsize_t;
            }

            H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                c_ptrTo(offset[0]), nil,
                                c_ptrTo(count[0]),  nil);

            var memspace = H5Screate_simple(2, c_ptrTo(count[0]), nil);
            ref localX   = data.X.localSlice[dom];
            H5Dread(dset, getHDF5Type(real(32)), memspace, filespace,
                    H5P_DEFAULT, c_ptrTo(localX));
            H5Sclose(memspace);
          }

          H5Sclose(filespace);
          H5Dclose(dset);
        }

        // ---- y ----------------------------------------------------------
        {
          var dset      = H5Dopen(fid, "y".c_str(), H5P_DEFAULT);
          var filespace = H5Dget_space(dset);

          for dom in data.y.localSubdomains() {
            var offset: [0..0] hsize_t = [dom.low:  hsize_t];
            var count:  [0..0] hsize_t = [dom.size: hsize_t];

            H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                c_ptrTo(offset[0]), nil,
                                c_ptrTo(count[0]),  nil);

            var memspace = H5Screate_simple(1, c_ptrTo(count[0]), nil);
            ref localY   = data.y.localSlice[dom];
            H5Dread(dset, getHDF5Type(real(32)), memspace, filespace,
                    H5P_DEFAULT, c_ptrTo(localY));
            H5Sclose(memspace);
          }

          H5Sclose(filespace);
          H5Dclose(dset);
        }

        H5Fclose(fid);
      }
    }

    return data;
  }

} // module HDF5Reader

import logging

import torch
from tensorflow.keras.callbacks import EarlyStopping, History, ModelCheckpoint

EarlyStopping = EarlyStopping
History = History


class ModelCheckpointTorch(ModelCheckpoint):
    def on_epoch_end(self, epoch: int, logs: dict = None):
        """
        Callback on epoch end. override to save models in torch.
        :param epoch: current epoch number
        :param logs: logs info
        """
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        file_path = self.filepath.format(epoch=epoch + 1, **logs)

        if self.save_best_only:
            self._save_best_model(epoch, file_path, logs)
        else:
            self._save_model(epoch, file_path)

    def _save_best_model(self, epoch: int, file_path: str, logs: dict = None):
        """
        Save a best model
        :param epoch: current epoch number
        :param file_path: path of file to save
        :param logs: logs info
        """
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                "Can save best model only with %s available, skipping." % self.monitor
            )
            return
        if not self.monitor_op(current, self.best):
            if self.verbose > 0:
                logging.info(
                    "Epoch %05d: %s did not improve from %0.5f"
                    % (epoch + 1, self.monitor, self.best)
                )
            return
        if self.verbose > 0:
            logging.info(
                "Epoch %05d: %s improved from %0.5f to %0.5f,"
                " saving model to %s"
                % (
                    epoch + 1,
                    self.monitor,
                    self.best,
                    current,
                    file_path,
                )
            )
        self.best = current
        if self.save_weights_only:
            torch.save(self.model.state_dict(), file_path)
        else:
            torch.save(self.model, file_path)

    def _save_model(self, epoch: int, file_path: str):
        """
        Save a model
        :param epoch: current epoch number
        :param file_path: path of file to save
        """
        if self.verbose > 0:
            logging.info("Epoch %05d: saving model to %s" % (epoch + 1, file_path))
        if self.save_weights_only:
            torch.save(self.model.state_dict(), file_path)
        else:
            torch.save(self.model, file_path)

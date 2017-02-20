namespace Retia.Training.Trainers
{
    public interface ITrainingStatusWriter
    {
        void UpdateEpochStatus(string status);
        void NewLine();
        void Message(string message);
    }
}